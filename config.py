import os
import torch
import yaml

config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open('./config.yaml', 'r', encoding='utf-8') as fp:  # contralearn/
    param = yaml.safe_load(fp)

# data_info
data_info = param['data_info']
data_dir = data_info['data_dir']
wav_dir = data_info['wav_dir']
model_dir = data_info['model_dir']
kmeans_dir = data_info['kmeans_dir']
feat_dir = data_info['feat_dir']
fs = data_info['fs']
n_mels = data_info['n_mels']
n_fft = data_info['n_fft']
hop_hength = data_info['hop_hength']
tar_time = data_info['tar_time']
patch_len = data_info['patch_len']
distance = data_info['distance']
CL_pos_mix_alpha = data_info['CL_pos_mix_alpha']

# man_control
man_control = param['man_control']
mix_way = man_control['mix_way']
choose_model_nums = man_control['choose_model_nums']
ext_way = man_control['ext_way']
zero_normal = man_control['zero_normal']
feat_normal = man_control['feat_normal']
clip_way = man_control['clip_way']
data_ext_way = man_control['data_ext_way']
model_name = man_control['model_name']
loss_mode =  man_control['loss_mode']
premodel_ext = man_control['premodel_ext']
learn_method = man_control['learn_method']
loss_choice = man_control['loss_choice']
kmean_draw3D = man_control['kmean_draw3D']
mix_up = man_control['mix_up']
kmean_needtrain = man_control['kmean_needtrain']
kmeans_needtest = man_control['kmeans_needtest']

# hyparameter
mach_index = param['mach_index']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cuda = param['cuda']
torch.cuda.device(cuda)  # torch.cuda.set_device(1)
mode = param['mode']
train_pickle = param['train_pickle']
test_pickle = param['test_pickle']
data_name = param['data_name']
cur_domain = param['cur_domain']
machID = param['machID']

# train
train = param['train']
input_dim = train['input_dim']
seed = train['seed']
new_seed = train['new_seed']
batch_size = train['batch_size']
MAX_EPOCH = train['epoch']
LR = train['lr']
weight_decay = train['weight_decay']
threshold_count = train['threshold_count']
loss_thre = train['loss_thre']
decimal_count = train['decimal_count']
threshold_alpha = train['threshold_alpha']
pauc_value = train['pauc_value']
num_workers = train['num_workers']
class_num = train['class_num']
detect_num = train['detect_num']
alpha = train['alpha']
global_pooling = train['global_pooling']
head_num = train['head_num']
head_size = train['head_size']
emb_dim = train['emb_dim']
mlp_hidden_size = train['mlp_hidden_size']
low_dim = train['low_dim']
temp = train['temp']
contra_loss_weight  = train['contra_loss_weight']

# kmeans
kmeans_model = param['kmeans_model']
# RandTimeShift
max_iter = kmeans_model['max_iter']
tol = kmeans_model['tol']

# da
da = param['da']
# RandTimeShift
do_rand_time_shift = da['do_rand_time_shift']
Tshift = da['Tshift']
# RandFreqShift
do_rand_freq_shift = da['do_rand_freq_shift']
Fshift = da['Fshift']
# RandomResizedCrop_diy
do_randcrop = da['do_randcrop']
rc_scale = tuple(da['rc_scale'])
rc_ratio = tuple(da['rc_ratio'])
# TimeReversal
do_time_reversal = da['do_time_reversal']
# Compander
do_compansion = da['do_compansion']
comp_alpha = da['comp_alpha']
# SpecAugment
do_time_warp = da['do_time_warp']
SpecAugment_W = da['SpecAugment_W']
do_freq_mask = da['do_freq_mask']
SpecAugment_F = da['SpecAugment_F']
SpecAugment_m_f = da['SpecAugment_m_f']
reduce_mask_range = da['reduce_mask_range']
do_time_mask = da['do_time_mask']
SpecAugment_T = da['SpecAugment_T']
SpecAugment_m_t = da['SpecAugment_m_t']
SpecAugment_mask_val = da['SpecAugment_mask_val']
# GaussNoise
awgn_stdev_gen = da['awgn_stdev_gen']
# RandomGaussianBlur
do_blur = da['do_blur']
blur_max_ksize = da['blur_max_ksize']
blur_stdev_x = da['blur_stdev_x']
