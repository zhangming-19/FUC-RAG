#######
# Subsample-aware and Back-Ex
#######

import copy
import os
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import numpy as np
import random
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
from da.data_augment import SpecAugment, RandomGaussianBlur, GaussNoise, RandTimeShift, RandFreqShift, TimeReversal, Compander
from da.rnd_resized_crop import RandomResizedCrop_diy
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import config as cfg


def plot_and_save_mel_spectrogram(audio_file, sr=16000, n_mels=128, hop_length=512, save_path=None, savename=None):
    if isinstance(audio_file, str):
        y, sr = librosa.load(audio_file, sr=sr)
    else:
        y = audio_file
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spectrogram_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Mel spectrogram saved to {save_path}")
    else:
        plt.show()
    plt.close()

def show_spec(log_spectrogram, titlename, savename):
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    from matplotlib.ticker import MultipleLocator
    rcParams['font.family'] = 'Times New Roman'
    rcParams['font.size'] = 12
    if isinstance(log_spectrogram, torch.Tensor):
        log_spectrogram = log_spectrogram.cpu().numpy()
    plt.figure(figsize=(5, 4), dpi=300)
    plt.title(titlename)
    librosa.display.specshow(log_spectrogram, x_axis='time', y_axis='mel', cmap='viridis', n_fft=1024,hop_length=512,sr=cfg.fs)
    ax = plt.gca() 
    ax.set_xlabel(f'Time(s)', fontsize=14)
    ax.set_ylabel(f'Frequency(Hz)', fontsize=14)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    #plt.colorbar(label='Log Amplitude')
    plt.tight_layout()
    plt.show()
    plt.savefig(f'./draw/S-{savename}', bbox_inches='tight')
    plt.close()

class DataAug(Dataset):
    def __init__(self, ):
        ### CutMix,mixup,gaussian,specaug
        super(DataAug, self).__init__()
        self.seed = cfg.seed
        np.random.seed(cfg.seed)
        random.seed(self.seed)

    # cutmix
    def rand_bbox(self, size, lam):
        W = size[3]
        H = size[2]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2


    def cutmix_batch(self, x, beta=1):
        B = x.size(0)
        x_anchor = x.clone()
        x_positive = x.clone()
        index = torch.randperm(B)

        for i in range(B):
            lam = np.random.beta(beta, beta)
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)

            x_positive[i, :, bby1:bby2, bbx1:bbx2] = x[index[i], :, bby1:bby2, bbx1:bbx2]

        return x_positive

    ### mixup
    def mixup(self, x, alpha=0.5):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1.0

        batch_size = x.size()[0]
        index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]

        return mixed_x

    ### gaussian noise
    def randomgaussian_batch(self, x, max_ksize=9, stdev_x=20):  # max_ksize=5
        x_np = x.squeeze(1).cpu().numpy()  # B, W, H
        blurred = []
        for i in range(x_np.shape[0]):
            kernel_size = tuple(2 * np.random.randint(0, max_ksize//2 + 1, 2) + 1)
            blurred_img = cv2.GaussianBlur(x_np[i], kernel_size, stdev_x)
            blurred.append(torch.tensor(blurred_img))
        blurred = torch.stack(blurred).unsqueeze(1).to(x.device)  # B, 1, W, H
        return blurred

    
    def spec_augment_batch(self, x, F=20, T=20, m_f=1, m_t=1, mask_val='zero', reduce_mask_range=0):
        B, C, W, H = x.shape
        x_aug = x.clone()

        for i in range(B):
            x_aug[i, 0] = self.freq_mask(x_aug[i, 0], F=F, n_masks=m_f, mask_val=mask_val, reduce_mask_range=reduce_mask_range)
            x_aug[i, 0] = self.time_mask(x_aug[i, 0], T=T, n_masks=m_t, mask_val=mask_val)

        return x_aug

    
    def freq_mask(self, _spec, F=30, n_masks=1, mask_val='zero', reduce_mask_range=0):
        W, H = _spec.shape
        for i in range(n_masks):
            bw = int(np.random.uniform(low=0.0, high=F))
            if reduce_mask_range == 0:
                f0 = random.randint(0, H - bw)
            else:
                f0 = random.randint(0, int(H * reduce_mask_range) - bw)

            if mask_val == 'zero':
                _spec[:, f0:f0 + bw] = 0
            elif mask_val == 'min':
                _spec[:, f0:f0 + bw] = _spec.min()
            elif mask_val == 'mean':
                _spec[:, f0:f0 + bw] = _spec.mean()
            elif mask_val == 'max':
                _spec[:, f0:f0 + bw] = _spec.max()
            elif mask_val == 'noise':
                _spec[:, f0:f0 + bw] = np.random.normal(_spec.mean(), _spec.std(), size=(W, bw))
        return _spec

    def time_mask(self, _spec, T=40, n_masks=1, mask_val='zero'):
        W, H = _spec.shape
        for i in range(n_masks):
            deltat = int(np.random.uniform(low=0.0, high=T))
            t0 = random.randint(0, W - deltat)

            if mask_val == 'zero':
                _spec[t0: t0 + deltat, :] = 0
            elif mask_val == 'min':
                _spec[t0: t0 + deltat, :] = _spec.min()
            elif mask_val == 'mean':
                _spec[t0: t0 + deltat, :] = _spec.mean()
            elif mask_val == 'max':
                _spec[t0: t0 + deltat, :] = _spec.max()
            elif mask_val == 'noise':
                _spec[t0: t0 + deltat, :] = np.random.normal(_spec.mean(), _spec.std(), size=(deltat, H))
        return _spec


class MyDataset(Dataset):

    def __init__(self, mode=cfg.mode, train_pickle=cfg.train_pickle, test_pickle=cfg.test_pickle):
        super(MyDataset, self).__init__()
        self.seed = cfg.seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.mode = mode
        self.train_pickle = train_pickle
        self.test_pickle = test_pickle
        self.cur_pickle = {'train': self.train_pickle, 'valid': self.train_pickle, 'test': self.test_pickle}
        self.load_data = self.load_phase()

    def load_phase(self):
        feat_path = os.path.join(cfg.feat_dir, self.cur_pickle[self.mode])
        features = joblib.load(feat_path)
        return features

    def split_phase(self, data):
        train_cs_data, valid_cs_data = train_test_split(data, test_size=0.2, random_state=self.seed)
        return train_cs_data, valid_cs_data

    def data_choose(self, mode=cfg.mode, train_data=None, valid_data=None, test_data=None):
        feats = {'train': train_data, 'valid': valid_data} if mode in ['train', 'valid'] else {'test':test_data}
        self.data = feats[mode]
        return self.data

    def __getitem__(self, index):
        feat_tensor, label, detect = self.data[index]
        return feat_tensor, label, detect

    def __len__(self):
        return len(self.data)


class AugContra():
    def __init__(self, mode=cfg.mode, data_name=cfg.data_name, seed = cfg.seed, data_ext_way=cfg.data_ext_way, clip_way=cfg.clip_way,
                 patch_len=cfg.patch_len, distance=cfg.distance, CL_pos_mix_alpha=cfg.CL_pos_mix_alpha, mix_way=cfg.mix_way,spec_draw=False, 
                 machID='', train_pickle=cfg.train_pickle,test_pickle=cfg.test_pickle, ext_way=cfg.ext_way, finalNorm=True):
        super(AugContra, self).__init__()
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        self.mode = mode
        self.train_pickle = train_pickle
        self.test_pickle = test_pickle
        self.ext_way = ext_way
        self.machID = machID
        self.data_name = data_name
        self.mean_stds = self.get_scalar()
        self.data_ext_way = data_ext_way
        self.patch_len = patch_len
        self.distance = distance
        self.CL_pos_mix_alpha = CL_pos_mix_alpha
        self.clip_way = clip_way
        self.mix_way = mix_way
        if self.data_ext_way == 'contraaugN':
            self.train_transform = self.load_train_augway()
        self.spec_draw = spec_draw

    def compute_rms_energy(self,signal):
        return np.sqrt((signal ** 2).mean())
    
    def imp_mixback(self,nowclip,backclip):
        now_E = self.compute_rms_energy(nowclip)
        back_E = self.compute_rms_energy(backclip)
        output_clip = (1-self.CL_pos_mix_alpha)*nowclip + self.CL_pos_mix_alpha*(now_E/back_E)*backclip
        return output_clip
    
    # train mode
    def crop_dualgram_train(self, load_data):
        if self.data_ext_way == 'rawclip':
            new_load_data = self.crop_dualgram_test(load_data=load_data)
        else:  #elif self.data_ext_way == 'contra':
            new_load_data = []
            backs = random.choices(load_data, k=2 * len(load_data))
            for idx, (now_feat, now_label, now_detect) in enumerate(load_data):
                now_row, now_col = now_feat.shape[0], now_feat.shape[1]
                back_feat1, back_feat2 = backs[2*idx][0], backs[2*idx +1][0]
                last_row1, last_col1 = back_feat1.shape[0], back_feat1.shape[1]
                last_row2, last_col2 = back_feat2.shape[0], back_feat2.shape[1]
                row, col = min([now_row, last_row1, last_row2]), min([now_col, last_col1, last_col2])
                
                if self.spec_draw is True:
                    show_spec(now_feat, f'Raw Log-Mel Spectrogram', f'rawspec.png')
                    show_spec(back_feat1, f'Background 1 Log-Mel Spectrogram', f'back1spec.png')
                    show_spec(back_feat2, f'Background 2 Log-Mel Spectrogram', f'back2spec.png')

                ### clip
                if self.clip_way == 'Random':
                    star_min_clip1 = 0
                    star_max_clip1 = col-self.patch_len

                    start_clip1 = random.randint(star_min_clip1,star_max_clip1)
                    start_clip2 = random.randint(star_min_clip1,star_max_clip1)
                    
                    now_clip1 = now_feat[:,start_clip1 : start_clip1 + self.patch_len]
                    now_clip2 = now_feat[:,start_clip2 : start_clip2 + self.patch_len]

                    back_clip1 = back_feat1[:,start_clip1 : start_clip1 + self.patch_len]
                    back_clip2 = back_feat2[:,start_clip2 : start_clip2 + self.patch_len]


                ### clip
                elif self.clip_way == 'Distance':
                    all_len = self.patch_len +self.distance
                    star_min_clip1 = 0
                    star_max_clip1 = col-all_len
                    start_clip1 = random.randint(star_min_clip1,star_max_clip1)

                    #print(start_clip1)

                    now_clip1 = now_feat[:,start_clip1:start_clip1+self.patch_len]
                    now_clip2 = now_feat[:,start_clip1+self.distance: start_clip1+self.distance + self.patch_len]  

                    back_clip1 = back_feat1[:,start_clip1:start_clip1+self.patch_len]
                    back_clip2 = back_feat2[:,start_clip1+self.distance: start_clip1+self.distance + self.patch_len]    

                    ###draw
                    if self.spec_draw is True:
                        show_spec(now_clip1, f'Clip 1 - Raw Log-Mel Spectrogram', f'rawClip1spec.png')
                        show_spec(now_clip2, f'Clip 2 - Raw Log-Mel Spectrogram', f'rawClip2spec.png')
                        show_spec(back_clip1, f'Clip 1 - Background Log-Mel Spectrogram', f'backClip1spec.png')
                        show_spec(back_clip2, f'Clip 2 - Background Log-Mel Spectrogram', f'backClip2spec.png')

                # mix
                if self.mix_way == 'linear':
                    new_clip1 = now_clip1 * (1-self.CL_pos_mix_alpha) + back_clip1 * self.CL_pos_mix_alpha
                    new_clip2 = now_clip2 * (1-self.CL_pos_mix_alpha) + back_clip2 * self.CL_pos_mix_alpha

                    ###draw
                    if self.spec_draw is True:
                        show_spec(new_clip1, f'Back-Ex Clip 1 - Log-Mel Spectrogram', f'MixClip1spec.png')
                        show_spec(new_clip2, f'Back-Ex Clip 2 - Log-Mel Spectrogram', f'MixClip2spec.png')

                elif self.mix_way == 'energy':
                    new_clip1 = self.imp_mixback(now_clip1,back_clip1)
                    new_clip2 = self.imp_mixback(now_clip2,back_clip2)

                # add
                new_load_data.append(((new_clip1, new_clip2), now_label, now_detect))

                if self.spec_draw is True:
                    break

        return new_load_data
    
    # test mode
    def crop_dualgram_test(self, load_data):  
        new_load_data = []
        for (now_feat, now_label, now_detect) in load_data:
            #show_spec(now_feat, './raw_spec.png')
            row, col = now_feat.shape[0], now_feat.shape[1]

            ### clip
            if self.clip_way == 'Random':
                star_min_clip1 = 0
                star_max_clip1 = col-self.patch_len

                start_clip1 = random.randint(star_min_clip1,star_max_clip1)
                start_clip2 = random.randint(star_min_clip1,star_max_clip1)

                now_clip1 = now_feat[:,start_clip1 : start_clip1 + self.patch_len]
                now_clip2 = now_feat[:,start_clip2 : start_clip2 + self.patch_len]

            elif self.clip_way == 'Distance':
                all_len = self.patch_len +self.distance
                star_min_clip1 = 0
                star_max_clip1 = col-all_len
                start_clip1 = random.randint(star_min_clip1,star_max_clip1)

                now_clip1 = now_feat[:,start_clip1:start_clip1+self.patch_len]
                now_clip2 = now_feat[:,start_clip1+self.distance: start_clip1+self.distance + self.patch_len]  

            #show_spec(now_clip1, './now_clip1_spec.png')
            #show_spec(now_clip2, './now_clip2_spec.png')
            new_load_data.append(((now_clip1, now_clip2), now_label, now_detect))
            #break

        return new_load_data


    # train mode
    def Mixbackcrop_dualgram_train(self, load_data):
        if self.data_ext_way == 'rawclip':
            new_load_data = self.crop_dualgram_test(load_data=load_data)
        else:  #lif self.data_ext_way == 'contra':
            new_load_data = []
            backs = random.choices(load_data, k=1 * len(load_data))
            for idx, (now_feat, now_label, now_detect) in enumerate(load_data):
                now_row, now_col = now_feat.shape[0], now_feat.shape[1]
                back_feat1 = backs[idx][0]
                last_row1, last_col1 = back_feat1.shape[0], back_feat1.shape[1]
                row, col = min([now_row, last_row1]), min([now_col, last_col1])

                ### clip
                if self.clip_way == 'Random':
                    star_min_clip1 = 0
                    star_max_clip1 = col-self.patch_len

                    start_clip1 = random.randint(star_min_clip1,star_max_clip1)
                    start_clip2 = random.randint(star_min_clip1,star_max_clip1)
                    
                    now_clip1 = now_feat[:,start_clip1 : start_clip1 + self.patch_len]
                    now_clip2 = now_feat[:,start_clip2 : start_clip2 + self.patch_len]

                    back_clip1 = back_feat1[:,start_clip1 : start_clip1 + self.patch_len]
                    back_clip2 = back_feat1[:,start_clip2 : start_clip2 + self.patch_len]


                ### clip
                elif self.clip_way == 'Distance':
                    all_len = self.patch_len +self.distance
                    star_min_clip1 = 0
                    star_max_clip1 = col-all_len
                    start_clip1 = random.randint(star_min_clip1,star_max_clip1)

                    #print(start_clip1)

                    now_clip1 = now_feat[:,start_clip1:start_clip1+self.patch_len]
                    now_clip2 = now_feat[:,start_clip1+self.distance: start_clip1+self.distance + self.patch_len]  

                    back_clip1 = back_feat1[:,start_clip1:start_clip1+self.patch_len]
                    back_clip2 = back_feat1[:,start_clip1+self.distance: start_clip1+self.distance + self.patch_len]    

                    ###draw
                    if self.spec_draw is True:
                        show_spec(now_clip1, f'Clip 1 - Raw Log-Mel Spectrogram', f'rawClip1spec.png')
                        show_spec(now_clip2, f'Clip 2 - Raw Log-Mel Spectrogram', f'rawClip2spec.png')
                        show_spec(back_clip1, f'Clip 1 - Background Log-Mel Spectrogram', f'backClip1spec.png')
                        show_spec(back_clip2, f'Clip 2 - Background Log-Mel Spectrogram', f'backClip2spec.png')

                # mix
                if self.mix_way == 'linear':
                    new_clip1 = now_clip1 * (1-self.CL_pos_mix_alpha) + back_clip1 * self.CL_pos_mix_alpha
                    new_clip2 = now_clip2 * (1-self.CL_pos_mix_alpha) + back_clip2 * self.CL_pos_mix_alpha

                    ###draw
                    if self.spec_draw is True:
                        show_spec(new_clip1, f'Back-Aug Clip 1 - Log-Mel Spectrogram', f'MixClip1spec.png')
                        show_spec(new_clip2, f'Back-Aug Clip 2 - Log-Mel Spectrogram', f'MixClip2spec.png')

                elif self.mix_way == 'energy':
                    new_clip1 = self.imp_mixback(now_clip1,back_clip1)
                    new_clip2 = self.imp_mixback(now_clip2,back_clip2)

                # add
                new_load_data.append(((new_clip1, new_clip2), now_label, now_detect))

                if self.spec_draw is True:
                    break

        return new_load_data
    
    # test mode
    def Mixbackcrop_dualgram_test(self, load_data):  
        new_load_data = []
        for (now_feat, now_label, now_detect) in load_data:
            #show_spec(now_feat, './raw_spec.png')
            row, col = now_feat.shape[0], now_feat.shape[1]

            ### clip
            if self.clip_way == 'Random':
                star_min_clip1 = 0
                star_max_clip1 = col-self.patch_len

                start_clip1 = random.randint(star_min_clip1,star_max_clip1)
                start_clip2 = random.randint(star_min_clip1,star_max_clip1)

                now_clip1 = now_feat[:,start_clip1 : start_clip1 + self.patch_len]
                now_clip2 = now_feat[:,start_clip2 : start_clip2 + self.patch_len]

            elif self.clip_way == 'Distance':
                all_len = self.patch_len +self.distance
                star_min_clip1 = 0
                star_max_clip1 = col-all_len
                start_clip1 = random.randint(star_min_clip1,star_max_clip1)

                now_clip1 = now_feat[:,start_clip1:start_clip1+self.patch_len]
                now_clip2 = now_feat[:,start_clip1+self.distance: start_clip1+self.distance + self.patch_len]  
            
            #show_spec(now_clip1, './now_clip1_spec.png')
            #show_spec(now_clip2, './now_clip2_spec.png')
            new_load_data.append(((now_clip1, now_clip2), now_label, now_detect))
            #break

        return new_load_data
    

    def load_data(self):
        if self.mode == 'train':
            dataset = MyDataset(mode='train', train_pickle=self.train_pickle)
            train_c_data, valid_c_data = dataset.split_phase(data=dataset.load_data)
            train_data = dataset.data_choose(mode='train', train_data=train_c_data)
            valid_data = dataset.data_choose(mode='valid', valid_data=valid_c_data)
            return train_data, valid_data
        elif self.mode == 'test':
            dataset = MyDataset(mode='test', test_pickle=self.test_pickle)
            test_data = dataset.data_choose(mode='test', test_data=dataset.load_data)
            return test_data

    def get_scalar(self):
        class_scalers_path = f'{cfg.feat_dir}/{self.ext_way}{self.machID}_scaler.pkl'
        with open(class_scalers_path, 'rb') as f:
            class_scalers = pickle.load(f)
        mean_stds = {}
        for label_idx,class_scaler in class_scalers.items():
            mean = class_scaler.mean_
            std = class_scaler.scale_  
            mean_stds[label_idx] = (mean, std)
        return mean_stds

    def load_train_augway(self, mean=None, std=None):
        train_transform = transforms.Compose([
            RandTimeShift(do_rand_time_shift=cfg.do_rand_time_shift, Tshift=cfg.Tshift),
            RandFreqShift(do_rand_freq_shift=cfg.do_rand_freq_shift, Fshift=cfg.Fshift),
            RandomResizedCrop_diy(do_randcrop=cfg.do_randcrop, scale=cfg.rc_scale,
                                ratio=cfg.rc_ratio),
            transforms.RandomApply([TimeReversal(do_time_reversal=cfg.do_time_reversal)], p=0.5),
            Compander(do_compansion=cfg.do_compansion, comp_alpha=cfg.comp_alpha),
            SpecAugment(do_time_warp=cfg.do_time_warp, W=cfg.SpecAugment_W,
                        do_freq_mask=cfg.do_freq_mask, F=cfg.SpecAugment_F, m_f=cfg.SpecAugment_m_f,
                        reduce_mask_range=cfg.reduce_mask_range,
                        do_time_mask=cfg.do_time_mask, T=cfg.SpecAugment_T, m_t=cfg.SpecAugment_m_t,
                        mask_val=cfg.SpecAugment_mask_val),
            GaussNoise(stdev_gen=cfg.awgn_stdev_gen),
            RandomGaussianBlur(do_blur=cfg.do_blur, max_ksize=cfg.blur_max_ksize,
                            stdev_x=cfg.blur_stdev_x),
            #transforms.ToTensor(),
            #transforms.Normalize(mean, std),
        ])

        return train_transform
    
    def load_finalNorm(self, mean=None, std=None):
        normal_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return normal_transform
    
    def apply_method(self, cur_data):
        new_cur_data = []
        for (feat1,feat2), label,detect in cur_data:
            if self.data_ext_way == 'contraaugN':
                feat1 = self.train_transform(feat1)
                feat2 = self.train_transform(feat2)            
            (cur_mean, cur_std) = self.mean_stds[label]
            feat1 = np.expand_dims(feat1.T, axis=0)
            feat2 = np.expand_dims(feat2.T, axis=0)
            if finalNorm is True:
                normal_transform = self.load_finalNorm(mean=cur_mean, std=cur_std)
                feat1 = normal_transform(feat1).squeeze()  #.permute(1, 0, 2)
                feat2 = normal_transform(feat2).squeeze()  #.permute(1, 0, 2)
            ###draw
            if self.spec_draw is True:
                show_spec(feat1, f'Normalized Back-Ex Clip 1 - Log-Mel Spectrogram', f'NorMixClip1spec.png')
                show_spec(feat2, f'Normalized Back-Ex Clip 2 - Log-Mel Spectrogram', f'NorMixClip2spec.png')
                break

            new_cur_data.append(((feat1,feat2), label, detect))
        return new_cur_data

    def data_save(self,traindata=None,validdata=None,testdata=None,Augfeat_path=None):
        if self.mode == 'train':  # 'raw', 'rawclip', 'contra','contraN','contraaugN'
            joblib.dump((traindata, validdata), Augfeat_path, compress=3)
        elif self.mode == 'test':
            joblib.dump(testdata, Augfeat_path, compress=3)

    def data_ContraAug(self, trainvalid='trainvalid'):
        Augfeat_path = os.path.join(cfg.feat_dir, f'{self.mode}_{self.data_name}_{self.data_ext_way}.joblib')
        #print(Augfeat_path)
        if self.mode == 'train':
            if os.path.exists(Augfeat_path):
                if trainvalid == 'trainvalid':
                    (train_data, valid_data) = joblib.load(Augfeat_path)
                    return train_data, valid_data
                else:
                    valid_data = joblib.load(Augfeat_path)
                    return valid_data
            else:
                print('No data!!!')
                train_data, valid_data = self.load_data()
                if self.data_ext_way != 'raw':
                    train_data = self.crop_dualgram_train(train_data)
                    valid_data = self.crop_dualgram_test(valid_data)
                if self.data_ext_way in ['contraN', 'contraaugN',]:
                    #cur_Augfeat_path = os.path.join(cfg.feat_dir, f'{self.mode}_{self.data_name}_contra.joblib')
                    #if os.path.exists(cur_Augfeat_path):
                        #(train_data, valid_data) = joblib.load(cur_Augfeat_path)
                    train_data = self.apply_method(train_data)
                    valid_data = self.apply_method(valid_data)
                self.data_save(traindata=train_data, validdata=valid_data,Augfeat_path=Augfeat_path)
                return train_data, valid_data

        elif self.mode == 'test':
            if os.path.exists(Augfeat_path):
                test_data = joblib.load(Augfeat_path)
                return test_data
            else:
                print('No data!!!')
                test_data  = self.load_data()
                if self.data_ext_way != 'raw':
                    test_data = self.crop_dualgram_test(test_data)  # Mixbackcrop_dualgram_test  crop_dualgram_test
                if self.data_ext_way in ['contraN', 'contraaugN',]:
                    #cur_Augfeat_path = os.path.join(cfg.feat_dir, f'{self.mode}_{self.data_name}_contra.joblib')
                    #if os.path.exists(cur_Augfeat_path):
                        #test_data = joblib.load(cur_Augfeat_path)
                    test_data = self.apply_method(test_data)  # train_allfeats
                # save
                self.data_save(testdata=test_data,Augfeat_path=Augfeat_path)
                return test_data


if __name__ == '__main__':
    spec_draw = False  # True False
    finalNorm = True
    #augcontra = AugContra()
    #outdata = augcontra.data_ContraAug_ALLGEN()
