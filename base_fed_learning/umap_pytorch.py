import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns

import umap
import pickle

#https://umap-learn.readthedocs.io/en/latest/transform.html
#https://github.com/lmcinnes/umap/blob/master/examples/mnist_transform_new_data.py
#kuzushiji (KMNIST)
#https://jlmelville.github.io/uwot/umap-examples.html
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ----------------------------------
# Initialization
# ----------------------------------
data_root_dir = '../data'
model_path_root = "./model_files/"

# ---------------------
# Check GPU
# ---------------------
torch.cuda.is_available()
device = torch.device("cuda:0")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('runing on GPU')
else:
    device = torch.device("cpu")
    print('runing on CPU')
    
torch.cuda.device_count()
# ----------------------------------
# Prepare data 
# ----------------------------------
# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# download the training and test datasets
dataset = datasets.MNIST(root=data_root_dir, train=True,
                                   download=True, transform=transform)
train_data, val_data = torch.utils.data.random_split(dataset, [50000, 10000])

# Create training and test dataloaders
num_workers = 0
batch_size = 1

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=False)

# specify the image classes
classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

# This is a semi-supervised setting and labels are of no immediate use except for FL part
train_data_list = [data[0] for data in train_loader]
train_data_tensor = torch.cat(train_data_list, dim=0)
train_labels_tensor = dataset.targets[train_data.indices]

print('umap embedding')
sns.set(context="paper", style="white")

reducer = umap.UMAP(random_state=42)
#####################
N = len(train_data_list)
data_mat = train_data_list[0].numpy()[0][0].flatten()
b = train_labels_tensor.numpy()[0:N]
for i in range(1,N):
    if i % (N//10) == 0:
        print("progress", i*100//N, " %")
    data_mat= np.vstack((data_mat, train_data_list[i].numpy()[0][0].flatten()))

embedding = reducer.fit_transform(data_mat)
pickle.dump( embedding, open( "umap_embedding.p", "wb" ) )
pickle.dump( reducer, open( "umap_reducer.p", "wb" ) )

fig, ax = plt.subplots(figsize=(12, 10))
color = b.astype(int)
plt.scatter(embedding[:, 0], embedding[:, 1], c=color, cmap="Spectral", s=0.1)
plt.setp(ax, xticks=[], yticks=[])
plt.title("MNIST data embedded into two dimensions by UMAP", fontsize=18)
