import numpy as np
import torch
from metrics import compute_iou, dice_coef_metric
from tqdm import tqdm
import copy

def train_model(model_name, model, dataloader, loss_func, optimizer, scheduler, measures, num_epochs, save_path='r2_unet.pt'):
    print('-'*10 , model_name, '-'*10)
    
    train_log = {k:[] for k in measures.keys()}
    train_log['loss'] = []
    val_log = {k:[] for k in measures.keys()}
    val_log['loss'] = []
    
    best_val_score = 0.
    best_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(num_epochs):
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_losses = []
            measurements = {k:0. for k in measures.keys()}

            for i, (images, masks) in enumerate(tqdm(dataloader[phase])):
                images = images.to(device)
                masks = masks.to(device)
                
                optimizer.zero_grad()

                outputs = model(images)
                loss = loss_func(outputs, masks)
                running_losses.append(loss.item())
        
                for (k,mobj) in measures.items():
                    measurements[k] += mobj(outputs, masks).item()
                
                if phase=='train':
                    loss.backward()
                    optimizer.step()
                           
            for k in measures.keys():
                measurements[k] = measurements[k] / len(dataloader[phase])
            measurements['loss'] = np.array(running_losses).mean()
            
            if phase=='val':
                scheduler.step(measurements['dice_coef'])
                
            if phase=='train':
                for k in measurements.keys():
                    train_log[k].append(measurements[k])
            else:
                for k in measurements.keys():
                    val_log[k].append(measurements[k])
                
            print(f'{phase}:', end='\t')
            for k,v in measurements.items():
                print(" {}:{:.4f}".format(k,v), end='  ')
            print()
            print(f'current lr:', optimizer.param_groups[0]['lr'])
            
            if phase=='val' and measurements['dice_coef'] > best_val_score:
                print(f"New score: {measurements['dice_coef']:.4f}\t Previous score: {best_val_score:.4f}")
                best_val_score = measurements['dice_coef']
                best_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), save_path)
                
    model.load_state_dict(best_wts)
    print(f'Training Completed. Best score: {best_val_score}')
    return train_log, val_log