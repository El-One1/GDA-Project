import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from PIL import Image
import pandas as pd
import os
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import numpy as np




def tsne_visualization(embeddings, labels, title="t-SNE Visualization"):

    
    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels)
    
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
    tsne_results = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='tab10', alpha=0.7, s=10)
    plt.colorbar(scatter, label='Classes')
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.show()


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

    exp_positive_sim = torch.exp(similarity_matrix) 
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

        # Build the full samples list
        # Map from image path to metadata row
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

        # Load the image
        img = Image.open(img_path).convert('RGB')

        # Apply transformations if provided
        if self.transform is not None:
            img = self.transform(img)
        
        return img, label, strata