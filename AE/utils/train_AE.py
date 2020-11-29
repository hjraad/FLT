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

def train_model(model, model_name, dataloaders, dataset_sizes, criterion, optimizer, scheduler, 
                num_epochs=10, model_save_dir='./', log_save_dir='./'):
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    least_loss = np.Inf
    
    with open(log_save_dir + model_name + ".log", "a") as f:
        for epoch in tqdm.tqdm(range(num_epochs)):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            temp_loss = {'train':0, 'test':0}
            for phase in ['train', 'test']:
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
                        outputs, _ = model(images)
                        loss = criterion(outputs, images)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    current_loss += loss.item() * images.size(0)
                    
                if phase == 'train':
                    scheduler.step()
                
                epoch_loss = current_loss / dataset_sizes[phase]
                temp_loss[phase] = epoch_loss

                print('{} Loss: {:.4f}'.format(phase, epoch_loss))

                # tag the best model
                if phase == 'test' and epoch_loss < least_loss:
                    least_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_epoch = epoch
                    
            # write a line for this epoch's loss values     
            f.write(f"{model_name},{round(time.time(),3)}, train_loss, {round(float(temp_loss['train']),4)}, test_loss, {round(float(temp_loss['test']),4)},{epoch}\n")
            
            print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Least test Acc: {:4f}, best epoch:{}'.format(least_loss, best_epoch))

    # save the best model
    torch.save({
            'epoch': best_epoch,
            'model_state_dict': model.state_dict(best_model_wts),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, model_save_dir + MODEL_NAME)
    
    return model, MODEL_NAME