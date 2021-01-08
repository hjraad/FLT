from __future__ import print_function


import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter


from six.moves import xrange

import umap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import pickle
import itertools
from sklearn.cluster import KMeans

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_data = datasets.CIFAR10(root="data", train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ]))

validation_data = datasets.CIFAR10(root="data", train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ]))

data_variance = np.var(training_data.data / 255.0)

    
class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)
    
class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self._conv_4 = nn.Conv2d(in_channels=num_hiddens, out_channels=num_hiddens//2, 
                                 kernel_size=1, stride=1)
        
        self._conv_5 = nn.Conv2d(in_channels=num_hiddens//2, out_channels=num_hiddens//16, 
                                 kernel_size=1, stride=1)
        
        self.fc1 = nn.Linear(8*8*num_hiddens//16, embedding_dim)
        

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        x = self._residual_stack(x)
        
        x = self._conv_4(x)
        x = F.relu(x)
        
        x = self._conv_5(x)
        x = F.relu(x)

        x = x.view(-1, 8*8*num_hiddens//16)
        x_comp = self.fc1(x)
        return x_comp

        
class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        
        self._linear_1 = nn.Linear(in_channels, 8*8*num_hiddens//16)
        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens//16, out_channels=num_hiddens//2, 
                                 kernel_size=1, stride=1)
        
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, out_channels=num_hiddens, 
                                 kernel_size=1, stride=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self._conv_trans_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        
        self._conv_trans_4 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        
        self._conv_trans_5 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=3,
                                                kernel_size=4, 
                                                stride=2, padding=1)

    def forward(self, inputs):
        
        x = self._linear_1(inputs)
        
        x = x.view(-1, num_hiddens//16, 8, 8)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        x = self._conv_trans_2(x)
        x = F.relu(x)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_3(x)
        x = F.relu(x)
        
        x = self._conv_trans_4(x)
        x = F.relu(x)
        
        return self._conv_trans_5(x)

batch_size = 256
num_training_updates = 20000

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 128

# commitment_cost = 0.25
# decay = 0.99

learning_rate = 1e-3

training_loader = DataLoader(training_data, 
                             batch_size=batch_size, 
                             shuffle=True,
                             pin_memory=True)
validation_loader = DataLoader(validation_data,
                               batch_size=32,
                               shuffle=True,
                               pin_memory=True)

class Model(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim):
        super(Model, self).__init__()
        
        self._encoder = Encoder(3, num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens, embedding_dim)
        
        self._decoder = Decoder(embedding_dim,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens)

    def forward(self, x):
        x_comp = self._encoder(x)
        x_recon = self._decoder(x_comp)

        return x_recon, x_comp
    
model = Model(num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

train_flag = False
eval_flag = False

if train_flag == True:
    model.train()
    train_res_recon_error = []
    train_res_perplexity = []

    for i in xrange(num_training_updates):
        (data, _) = next(iter(training_loader))
        data = data.to(device)
        optimizer.zero_grad()

        data_recon, data_comp = model(data)
        loss = F.mse_loss(data_recon, data) / data_variance
        loss.backward()

        optimizer.step()
        
        train_res_recon_error.append(loss.item())
        # train_res_perplexity.append(perplexity.item())

        if (i+1) % 100 == 0:
            print('%d iterations' % (i+1))
            print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
            # print('perplexity: %.3f' % np.mean(train_res_perplexity[-100:]))
            print()
        
    train_res_recon_error_smooth = savgol_filter(train_res_recon_error, 201, 7)

    f = plt.figure(figsize=(16,8))
    ax = f.add_subplot(1,2,1)
    ax.plot(train_res_recon_error_smooth)
    ax.set_yscale('log')
    ax.set_title('Smoothed NMSE.')
    ax.set_xlabel('iteration')


if eval_flag == True:
    model.eval()

    embedding_list = []
    labels_list = []
    with torch.no_grad():
        for _, (image, label) in enumerate(tqdm(training_data, desc='Inferencing training embedding')):
                image = image.to(device)
                labels_list.append(label) 
                _, embedding = model(image.unsqueeze(0))
                embedding_list.append(embedding.cpu().detach().numpy())
    ae_embedding_np = np.concatenate(embedding_list, axis=0)
    ae_labels_np = np.array(labels_list)
    pickle.dump((ae_embedding_np, ae_labels_np), open('./new_embedding.p', 'wb'))
else:
    (ae_embedding_np, ae_labels_np) = pickle.load(open('./new_embedding.p', 'rb'))

nr_clsuters_kmeans = 1
num_user = 10
centers = np.zeros((num_user*len(np.unique(ae_labels_np)), nr_clsuters_kmeans, 128))
for ind in range(len(np.unique(ae_labels_np))):
    selec_inds = np.where(ae_labels_np == ind)[0]
    np.random.shuffle(selec_inds)
    for i in range(num_user):
        selec_embedding = np.squeeze(ae_embedding_np[selec_inds[i*len(selec_inds)//num_user:(i+1)*len(selec_inds)//num_user],:])
        kmeans = KMeans(n_clusters=nr_clsuters_kmeans, random_state=43).fit(selec_embedding)
        centers[num_user*ind+i,:,:] = kmeans.cluster_centers_

# umap_reducer = umap.UMAP(n_components=2, random_state=43)
# umap_embedding = umap_reducer.fit_transform(np.reshape(centers, (-1, embedding_dim)))
umap_embedding = np.reshape(centers, (-1, embedding_dim))
# centers = np.reshape(umap_embedding, (len(np.unique(ae_labels_np)), -1, 2))


# sns.set(context='paper', style='white')
# nr_of_centers = nr_clsuters_kmeans*1
# colors = itertools.cycle(["r"]*nr_of_centers + 
#                          ["b"]*nr_of_centers + 
#                          ["g"]*nr_of_centers + 
#                          ["k"]*nr_of_centers + 
#                          ["y"]*nr_of_centers)

plt.figure()
classes = [str(nr) for nr in range(0, len(np.unique(ae_labels_np)))]
colors = np.repeat(np.arange(len(np.unique(ae_labels_np))), num_user)
plt.scatter(umap_embedding[:,0], umap_embedding[:,1], c=colors, 
                s=8, cmap='tab10', label=classes)
plt.show()
plt.savefig('./new_ae.jpg')