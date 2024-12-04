import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn

from torchvision import transforms

from PIL import Image
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import numpy as np




def tsne_visualization(embeddings, labels, stratum, title="t-SNE Visualization", save_path = None):

    
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)
    stratum = np.asarray(stratum)
    
    tsne_labels = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    tsne_results_labels = tsne_labels.fit_transform(embeddings)

    tsne_stratum = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    tsne_results_stratum = tsne_stratum.fit_transform(embeddings)

    plt.figure(figsize=(14, 7))

    plt.subplot(1, 2, 1)
    scatter_labels = plt.scatter(tsne_results_labels[:, 0], tsne_results_labels[:, 1], c=labels, cmap='tab10', alpha=0.7, s=10)
    plt.colorbar(scatter_labels, label='Classes')
    plt.title(f"{title} - Classes")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(1, 2, 2)
    scatter_stratum = plt.scatter(tsne_results_stratum[:, 0], tsne_results_stratum[:, 1], c=stratum, cmap='tab20', alpha=0.7, s=10)
    plt.colorbar(scatter_stratum, label='Strata')
    plt.title(f"{title} - Strata")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
    if save_path:
        plt.savefig(save_path)


def full_loss(features, labels, alpha=0.5, temperature = 0.5):

    ################## LSC LOSS ##################

    features = F.normalize(features, dim=1)  # Normalize features to lie on hypersphere

    similarity_matrix = torch.matmul(features, features.T) / temperature
    
    labels = labels.unsqueeze(1)
    positive_mask = (labels == labels.T).float()
    negative_mask = 1 - positive_mask

    exp_sim = torch.exp(similarity_matrix) * negative_mask
    log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
    positive_sim = (log_prob * positive_mask).sum(dim=1) / positive_mask.sum(dim=1)
    lsc_loss = -positive_sim.mean()

    ################## LSC LOSS ##################



    ################## LREP LOSS ##################

    exp_positive_sim = torch.exp(similarity_matrix) * positive_mask
    repel_loss_tempo = -torch.log(exp_positive_sim.diagonal() / exp_positive_sim.sum(dim=1))
    repel_loss = repel_loss_tempo.mean()

    ################## LREP LOSS ##################

    return alpha * lsc_loss + (1 - alpha) * repel_loss


class WaterbirdsFullData(Dataset):
    def __init__(self, root, metadata_csv, split, transform=None):

        self.root = root
        self.transform = transform
        self.split = 0 if split == 'train' else 1 if split == 'val' else 2

        self.metadata = pd.read_csv(metadata_csv)

        self.metadata = self.metadata[self.metadata['split'] == self.split]

        self.classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.idx_to_class = {i: self.classes[i] for i in range(len(self.classes))}


        self.samples = []
        for class_name, _ in self.class_to_idx.items():
            class_folder = os.path.join(root, class_name)
            for img_filename in os.listdir(class_folder):
                full_img_path = os.path.join(class_folder, img_filename)
                metadata_img_filename = os.path.join(class_name, img_filename)

                metadata_row = self.metadata[self.metadata['img_filename'] == metadata_img_filename]
                if not metadata_row.empty:
                    label = metadata_row['y'].values[0]
                    strata = metadata_row['place'].values[0]
                    self.samples.append((full_img_path, label, strata))
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            tuple: (image, class_label, strata)
        """
        img_path, label, strata = self.samples[idx]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        return img, label, strata




class CIFAR100CoarseUnbalanced(Dataset):

    
    def __init__(self, original_dataset):
        
        self.super_class_sub_class_correspondence = {4: [0, 51, 53, 57, 83],
                                                    1: [1, 32, 67, 73, 91],
                                                    14: [2, 11, 35, 46, 98],
                                                    8: [3, 42, 43, 88, 97],
                                                    0: [4, 30, 55, 72, 95],
                                                    6: [5, 20, 25, 84, 94],
                                                    7: [6, 7, 14, 18, 24],
                                                    18: [8, 13, 48, 58, 90],
                                                    3: [9, 10, 16, 28, 61],
                                                    9: [12, 17, 37, 68, 76],
                                                    11: [15, 19, 21, 31, 38],
                                                    5: [22, 39, 40, 86, 87],
                                                    10: [23, 33, 49, 60, 71],
                                                    13: [26, 45, 77, 79, 99],
                                                    15: [27, 29, 44, 78, 93],
                                                    12: [34, 63, 64, 66, 75],
                                                    16: [36, 50, 65, 74, 80],
                                                    19: [41, 69, 81, 85, 89],
                                                    17: [47, 52, 56, 59, 96],
                                                    2: [54, 62, 70, 82, 92]}
        
        self.super_class_sub_class_correspondence = {k: self.super_class_sub_class_correspondence[k] 
                                               for k in sorted(self.super_class_sub_class_correspondence.keys())}


        sub_classes_proportion_inside_superclasses = [500, 250, 100, 50, 50]


        self.data = np.zeros((20 * sum(sub_classes_proportion_inside_superclasses), 32, 32, 3))
        self.targets = np.zeros(20 * sum(sub_classes_proportion_inside_superclasses))
        self.original_targets = original_dataset.targets
        self.original_target_link = np.zeros(20 * sum(sub_classes_proportion_inside_superclasses))
        
        sub_classes_proportion_inside_superclasses = np.array(sub_classes_proportion_inside_superclasses)
        
        data_idx = 0  
        
        for superclass in range(20):
            for i, subclass in enumerate(self.super_class_sub_class_correspondence[superclass]):

                indices = np.where(np.array(self.original_targets) == subclass)[0]
                selected_indices = np.random.choice(indices, sub_classes_proportion_inside_superclasses[i], replace=False)
                
                for j, idx in enumerate(selected_indices):

                    final_idx = data_idx + j
                    
                    self.data[final_idx] = original_dataset.data[idx]
                    self.targets[final_idx] = superclass
                    self.original_target_link[final_idx] = subclass
                
                data_idx += sub_classes_proportion_inside_superclasses[i]
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        #transform np array to torch tensor and put (C, H, W) instead of (H, W, C), and normalize for imagenet stats
        imagenet_stats = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
        data = transforms.functional.to_tensor(self.data[idx] / 255)
        data = transforms.functional.normalize(data, *imagenet_stats)

        return data.float(), self.targets[idx], self.original_target_link[idx]
    
def sparse2coarse(targets):  ### source: https://github.com/ryanchankh/cifar100coarse/blob/master/sparse2coarse.py
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.

    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]


class CustomViT(nn.Module):
    def __init__(self, num_classes=10, patch_size=4, num_layers=7, embed_dim=256, num_heads=4, ff_dim=512):
        super(CustomViT, self).__init__()

        self.patch_embed = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=ff_dim, 
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)  # Shape: (batch_size, embed_dim, num_patches, 1)
        x = x.flatten(2).transpose(1, 2)  # Shape: (batch_size, num_patches, embed_dim)

        x = self.transformer(x)

        x = x.mean(dim=1)

        x = self.fc(x)
        return x