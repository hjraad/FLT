'''
Model training procedure
Author: Hadi Jamali-Rad
email: h.jamalirad@gmail.com
'''
import torch
import time
import tqdm
import copy
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, model_name, dataloaders, dataset_sizes, phases, criterion, optimizer, scheduler, 
                num_epochs=10, num_hiddens=128,  model_save_dir='./', log_save_dir='./', save_flag=True):
    '''
    Trains the AE model 
    Paramteres:
        model & model_name, 
        dataloaders: dictionary of train and test torchvision dataloaders
        datasetsize: dictionary of train and test datasets
        criterion: loss
        optimizer
        scheduler: learning rate scheduler
    Returns:
        best and last models  
    '''
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    least_loss = np.Inf
    
    with open(f'{log_save_dir}/{model_name}.log', 'a') as f:
        for epoch in tqdm.tqdm(range(num_epochs), desc='Training progress'):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            temp_loss = {phase:0 for phase in phases} 
            
            for phase in phases:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                current_loss = 0.0

                # Iterate over data.
                # semi-supervised => labels are unimportant
                for images, _ in dataloaders[phase]:
                    images = images.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs, _ = model(images, num_hiddens)
                        loss = criterion(outputs, images)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    current_loss += loss.item() * images.size(0)
                    
                if phase == 'train' and scheduler:
                    scheduler.step()
                
                epoch_loss = current_loss / dataset_sizes[phase]
                temp_loss[phase] = epoch_loss

                print('{} Loss: {:.4f}'.format(phase, epoch_loss))

                # tag the best model (to)
                if phase == 'test' and epoch_loss < least_loss:
                    least_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_opt_wts = copy.deepcopy(optimizer.state_dict())
                    best_model = copy.deepcopy(model)
                    best_epoch = epoch
                    
            # write a line for this epoch's loss values  
            if 'test' in phases:
                f.write(f"{model_name},{round(time.time(),3)}, train_loss, {round(float(temp_loss['train']),4)}, test_loss, {round(float(temp_loss['test']),4)},{epoch}\n")
            else:
                f.write(f"{model_name},{round(time.time(),3)}, train_loss, {round(float(temp_loss['train']),4)},{epoch}\n")
            print()
    
    # last model     
    last_model = copy.deepcopy(model)
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    if 'test' in phases:
        print('Least test Acc: {:4f}, best epoch:{}'.format(least_loss, best_epoch))
    else:
        best_model = None

    # save the best model
    if save_flag:
        if 'test' in phases:
            torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': best_model_wts,
                    'optimizer_state_dict': best_opt_wts,
                    'loss': least_loss,
                    }, f'{model_save_dir}/{model_name}_best.pt')
            print('saved the best model.')
        # save the last model
        torch.save({
                'epoch': num_epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                }, f'{model_save_dir}/{model_name}_last.pt')
        print('saved the last model.')
    
    return best_model, last_model