#######
# fusion loss, and paired subsample fusion anomaly score
#######

import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from config import *


class ModelTrainerContra(object):
    
    @staticmethod
    def compute_losses(embedi, embedj, labels, loss_fn, loss_contra, model_head, loss_mode, contra_loss_weight):
        if loss_mode == 'cls_':
            # Compute ArcFace loss for classification
            loss_arcface_i = loss_fn(embedi, labels)  # Subcon embedi.unsqueeze(1),ArcFace embedi
            loss_arcface_j = loss_fn(embedj, labels)
            return (loss_arcface_i + loss_arcface_j) / 2, loss_arcface_i, loss_arcface_j, None

        elif loss_mode == 'contra_':
            # Map embeddings with model_head for contrastive loss
            projected_i = model_head(embedi)  # Projected embedding i
            projected_j = model_head(embedj)  # Projected embedding j
            loss_contra_value = loss_contra(projected_i, projected_j)
            return loss_contra_value, None, None, loss_contra_value

        elif loss_mode == 'clscontra_':
            # Compute ArcFace loss for classification
            loss_arcface_i = loss_fn(embedi, labels)
            loss_arcface_j = loss_fn(embedj, labels)
            # Map embeddings with model_head for contrastive loss
            projected_i = model_head(embedi)  # Projected embedding i
            projected_j = model_head(embedj)  # Projected embedding j
            #print(projected_i.shape)
            #print(projected_j.shape)
            # Compute NTXent contrastive loss
            loss_contra_value = loss_contra(projected_i, projected_j)
            # Combine both losses
            combined_loss = (1 - contra_loss_weight) * (loss_arcface_i + loss_arcface_j) / 2 + contra_loss_weight * loss_contra_value
            return combined_loss, loss_arcface_i, loss_arcface_j, loss_contra_value

        return None, None, None, None  # Return None if no loss mode matched

    @staticmethod
    def train(data_loader, model, model_head, loss_fn, loss_contra, optimizer, epoch_id, device, max_epoch, loss_mode, contra_loss_weight, dataaug=None,augmethod=None):
        model.train()
        model_head.train()
        loss_fn.train()
        loss_contra.train()
        
        conf_mat = np.zeros((class_num, class_num))
        loss_sigma, arci, arcj, contra = [], [], [], []
        model_type = 'Training'
        progress_bar = tqdm(data_loader, total=len(data_loader), desc=f'{model_type:<10} (Epoch {epoch_id + 1}/{max_epoch})')

        for(ex1, ex2), labels, _ in progress_bar:

            ex1, ex2 = ex1.to(device, dtype=torch.float32).unsqueeze(1), ex2.to(device, dtype=torch.float32).unsqueeze(1)
            
            # dataaug
            if dataaug:
                if 'cutmix' in augmethod:
                    ex1 = dataaug.cutmix_batch(ex1)
                    ex2 = dataaug.cutmix_batch(ex2)
                elif 'mixup' in augmethod:
                    ex1 = dataaug.mixup(ex1)
                    ex2 = dataaug.mixup(ex2)
                elif 'gaussian' in augmethod:
                    ex1 = dataaug.randomgaussian_batch(ex1)
                    ex2 = dataaug.randomgaussian_batch(ex2)
                elif 'specaug' in augmethod:
                    ex1 = dataaug.spec_augment_batch(ex1)
                    ex2 = dataaug.spec_augment_batch(ex2)

            labels = labels.to(device, dtype=torch.long)

            # Get encoder output (h_i, h_j) to feed the projection head
            resulti, embedi = model(ex1)  # embedding i (for ex1)
            resultj, embedj = model(ex2)  # embedding j (for ex2)  resultj, embedj = model(ex2)
            #print(embedi.shape)
            #print(embedj.shape)

            losses, loss_arcface_i, loss_arcface_j, loss_contra_value = ModelTrainerContra.compute_losses(embedi, embedj, labels, loss_fn, loss_contra, model_head, loss_mode, contra_loss_weight)

            if loss_mode in ['cls_', 'clscontra_']:
                arci.append(loss_arcface_i.item())
                arcj.append(loss_arcface_j.item())

            if loss_mode in ['contra_', 'clscontra_']:
                contra.append(loss_contra_value.item())

            losses.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Log and accumulate losses
            loss_sigma.append(losses.item())

            # Predictions and metrics
            _, predicted = torch.max((resulti+resultj)/2, 1)
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].cpu().numpy()
                conf_mat[cate_i, pre_i] += 1.

            acc_avg = conf_mat.trace() / conf_mat.sum()
            progress_bar.set_postfix({'Loss': f'{np.mean(loss_sigma):.5f}', 'Acc': f'{acc_avg:.5f}', 'ArcI': f'{np.mean(arci):.6f}', 'ArcJ': f'{np.mean(arcj):.6f}', 'Contra': f'{np.mean(contra):.6f}'})
        progress_bar.close()
        return acc_avg, np.mean(loss_sigma)


# kmean - kmean_featext
def kmean_featext(test_loader, test_model, model):
    use_progress = False
    tarin_progress_bar = tqdm(test_loader, total=len(test_loader), desc=f'{test_model:<10}') if use_progress else test_loader
    test_newdata = []
    with torch.no_grad():
        for (ex, labels, detects) in tarin_progress_bar:
            ex = ex.to(device, dtype=torch.float32).unsqueeze(1)
            labels, detects = labels.to(device, dtype=torch.long), detects.to(device, dtype=torch.long)
            _, embedis = model(ex)  # embedding i (for ex1)
            for (embedi, label, detect) in zip(embedis, labels, detects):
                test_newdata.append((embedi, label, detect))
    return test_newdata

# kmean - kmean_featext_contra
def kmean_featext_contra(train_loader, test_model, model, ):
    tarin_newdata = []
    use_progress = False
    iterator = tqdm(train_loader, total=len(train_loader), desc=f'{test_model:<10}') if use_progress else train_loader

    with torch.no_grad():
        for ((ex1, ex2), labels, detects) in iterator:  # tarin_progress_bar
            ex1, ex2 = ex1.to(device, dtype=torch.float32).unsqueeze(1), ex2.to(device, dtype=torch.float32).unsqueeze(1)
            labels, detects = labels.to(device, dtype=torch.long), detects.to(device, dtype=torch.long)
            _, embedis = model(ex1)  # embedding i (for ex1)
            _, embedjs = model(ex2)  # embedding j (for ex2)
            for (embedi, embedj, label, detect) in zip(embedis, embedjs, labels, detects):
                tarin_newdata.append((embedi, label, detect))
                tarin_newdata.append((embedj, label, detect))
    return tarin_newdata


def kmeans_test(self, test_data):
    model_name = self.model_pklpath.split('_')[-2]
    test_feats = [[] for _ in range(class_num)]
    for embed, label, detect in test_data:
        test_feats[label].append((embed, label, detect))
    avg_auc_helps = []
    avg_pauc_helps = []
    per_class_results = {}

    for idx, test_feat in enumerate(test_feats):
        cur_model_pklpath = self.model_pklpath.replace('.pkl', f'_{idx}.pkl')
        kmeans_model = CustomKMeans.load_model(cur_model_pklpath)
        test_features = []
        test_real_detects = []
        for feat, label, detect in test_feat:
            test_features.append(feat)
            test_real_detects.append(detect)
        test_real_detects = np.asarray(test_real_detects)

        test_distances = kmeans_model._compute_distances(test_features)
        if self.learn_method == 'Contrastive':
            test_weighted_distances = [
                (d1 + d2) / 2
                for d1, d2 in zip(
                    test_distances[::2],
                    test_distances[1::2]
                )
            ]
            test_real_detects = test_real_detects[::2]
        else:
            test_weighted_distances = test_distances

        class_distances = np.asarray([
            torch.min(distance).cpu().numpy()
            for distance in test_weighted_distances
        ])

        assert set(test_real_detects).issubset({0, 1})
        class_auc = roc_auc_score(
            test_real_detects,
            class_distances
        )

        pauc = CustomKMeans.calculate_pauc(
            test_real_detects,
            class_distances
        )

        avg_auc_helps.append(class_auc)
        avg_pauc_helps.append(pauc)

        per_class_results[idx] = {
            "AUC": class_auc,
            "pAUC": pauc,
        }

    all_data_dict = {
        'model_name': model_name,
        'avg_auc_help': np.mean(avg_auc_helps),
        'avg_pauc_help': np.mean(avg_pauc_helps),

        **{
            f'class{label}_AUC':
            per_class_results[label]["AUC"]
            for label in per_class_results
        },

        **{
            f'class{label}_pAUC':
            per_class_results[label]["pAUC"]
            for label in per_class_results
        },
    }

    return (
        all_data_dict,
        per_class_results,
        np.mean(avg_auc_helps),
        np.mean(avg_pauc_helps)
    )


def load_testmodels(model_save):
    test_models = []
    model_loss_pairs = []
    for save_model_name in os.listdir(model_save):
        if save_model_name.endswith(".pth"):
            parts = save_model_name.split('_')
            loss_value = float(parts[-1].replace('.pth', ''))  # Extract loss from the last part and convert to float
            model_loss_pairs.append((loss_value, save_model_name))  # Store the model and its loss value
    model_loss_pairs.sort(key=lambda x: x[0])  # Sort models by loss values in ascending order
    for loss, model_filename in model_loss_pairs:  #  [:choose_model_nums] Select the top 10 models with the best loss
        path_ = os.path.join(model_save, model_filename)
        test_models.append((model_filename, path_))
    return test_models


def plot_valid_loss(train_loss_list, valid_loss_list):
    epochs = range(1, len(valid_loss_list) + 1)
    plt.plot(epochs, train_loss_list, 'r', label='Training loss')
    plt.plot(epochs, valid_loss_list, 'b', label='Validating loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    # plt.show()
    fig = plt.gcf()

    return fig

