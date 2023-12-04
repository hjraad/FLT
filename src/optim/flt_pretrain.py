#----------------------------------------------------------------------------
# Created By  : Mohammad Abdizadeh & Hadi Jamali-Rad
# Created Date: 23-Nov-2020
# 
# Refactored By: Sayak Mukherjee
# Last Update: 30-Nov-2023
# ---------------------------------------------------------------------------
# File contains the code for pretraining the feature extractor.
# ---------------------------------------------------------------------------

import logging
import torch

from pathlib import Path
from models.utils import save_model
from torch.utils.data import DataLoader
from datasets.load_dataset import load_dataset

logger = logging.getLogger(__name__)

class FLTPretrain():

    def __init__(self, config, extractor, model_name, pretrain_dataset, device):

        self.logger = logging.getLogger(self.__class__.__name__)

        self.config = config
        self.device = device
        self.net = extractor
        self.model_name = model_name
        self.pretrain_dataset = pretrain_dataset
        self.dataset, _, _= load_dataset(pretrain_dataset.upper(), config.dataset.path, dataset_split=config.dataset.dataset_split)

        self.batch_size = self.config.dataset.train_batch_size
        self.nr_epochs = self.config.trainer.pretrain_epochs

        self.model_path = Path(self.config.project.path)\
            .joinpath('flt_artifacts')
        
        if not Path.exists(self.model_path):
            Path.mkdir(self.model_path, parents=True)

    def train(self):
        """Training loop
        """
        self.net = self.net.to(self.device)
        
        # Set mode to train model
        self.net.train()

        # Set optimizer for the local updates
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)

        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        criterion = torch.nn.MSELoss().to(self.device)

        trainloader = DataLoader(self.dataset['train'], batch_size = self.batch_size)
        testloader = DataLoader(self.dataset['test'], batch_size = self.batch_size)

        min_loss = 0
        best_state_dict = None

        logger.info(f'Pre-training extractor on {self.pretrain_dataset} dataset.')

        for iter in range(self.nr_epochs):

            train_loss = []

            for batch_idx, (images, _) in enumerate(trainloader):

                images = images.to(self.device)

                self.net.zero_grad()
                x_recon, _ = self.net(images)
                loss = criterion(x_recon, images)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # exp_lr_scheduler.step()
                
                train_loss.append(loss.item())

            self.logger.info(f'Epoch {iter} | Train Loss: {sum(train_loss)/len(train_loss)}')

            with torch.no_grad():
                test_loss = [] 

                for batch_idx, (images, _) in enumerate(testloader):

                    images = images.to(self.device)

                    x_recon, _ = self.net(images)

                    loss = criterion(x_recon, images)
                    
                    test_loss.append(loss.item())

            self.logger.info(f'Epoch {iter} | Test Loss: {sum(test_loss)/len(test_loss)}')

            curr_loss = sum(test_loss)/len(test_loss)
            if iter == 0:
                min_loss = curr_loss
                best_state_dict = self.net.state_dict()
            else:
                if min_loss > curr_loss:
                    min_loss = curr_loss
                    best_state_dict = self.net.state_dict()

        self.net.load_state_dict(best_state_dict)

        model_save_path = self.model_path.joinpath(self.model_name + '_' + self.pretrain_dataset + '.tar')
        save_model(self.net, model_save_path)

        logger.info(f'Pre-trained model saved in {model_save_path}.')

        return self.net

    def finetune(self):
        """Training loop
        """

        model = self.net
        model = model.to(self.device)
        
        # Set mode to train model
        model.train()

        # Set optimizer for the local updates
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        
        criterion = torch.nn.MSELoss().to(self.device)

        trainloader = DataLoader(self.dataset['train'], batch_size = self.batch_size)
        testloader = DataLoader(self.dataset['test'], batch_size = self.batch_size)

        finetune_epochs = self.config.trainer.finetune_epochs
        min_loss = 0
        best_state_dict = None

        # performance before finetuning
        with torch.no_grad():
            test_loss = [] 

            for batch_idx, (images, _) in enumerate(testloader):

                images = images.to(self.device)

                x_recon, _ = model(images)

                loss = criterion(x_recon, images)
                
                test_loss.append(loss.item())

        self.logger.info(f'Before pre-training | Test Loss: {sum(test_loss)/len(test_loss)}')
        curr_loss = sum(test_loss)/len(test_loss)

        min_loss = curr_loss
        best_state_dict = model.state_dict()

        # start finetuning
        logger.info(f'Finetuning extractor on {self.pretrain_dataset} dataset.')

        for iter in range(finetune_epochs):

            train_loss = []

            for batch_idx, (images, _) in enumerate(trainloader):

                images = images.to(self.device)

                self.net.zero_grad()
                x_recon, _ = model(images)
                loss = criterion(x_recon, images)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                exp_lr_scheduler.step()
                
                train_loss.append(loss.item())

            self.logger.info(f'Epoch {iter} | Train Loss: {sum(train_loss)/len(train_loss)}')

            with torch.no_grad():
                test_loss = [] 

                for batch_idx, (images, _) in enumerate(testloader):

                    images = images.to(self.device)

                    x_recon, _ = model(images)

                    loss = criterion(x_recon, images)
                    
                    test_loss.append(loss.item())

            self.logger.info(f'Epoch {iter} | Test Loss: {sum(test_loss)/len(test_loss)}')

            curr_loss = sum(test_loss)/len(test_loss)
            if iter == 0:
                min_loss = curr_loss
                best_state_dict = model.state_dict()
            else:
                if min_loss > curr_loss:
                    min_loss = curr_loss
                    best_state_dict = self.net.state_dict()

        self.logger.info(f'Using model with test loss {min_loss}')
        model.load_state_dict(best_state_dict)

        return model