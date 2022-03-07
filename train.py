import numpy as np
import torch
from metrics import compute_iou, dice_coef_metric
from tqdm import tqdm
import copy

def train_model(model_name, model, train_loader, val_loader, loss_func, optimizer, scheduler, num_epochs, save_path='attn_unet.pt'):
    print('-'*10, model_name, '-'*10)

    loss_history = []
    train_history = []
    val_history = []
    
    best_val_score = 0.
    best_wts = copy.deepcopy(model.state_dict())
    
    for epoch in range(num_epochs):
        model.train()
        
        losses = []
        train_iou = []
        
        for i, (image, mask) in enumerate(tqdm(train_loader)):
            image = image.to(device)
            mask = mask.to(device)
            outputs = model(image)
            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < 0.5)] = 0.0
            out_cut[np.nonzero(out_cut >= 0.5)] = 1.0            
            
            train_dice = dice_coef_metric(out_cut, mask.data.cpu().numpy())
            loss = loss_func(outputs, mask)
            losses.append(loss.item())
            train_iou.append(train_dice)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
        val_mean_iou = compute_iou(model, val_loader)
        scheduler.step(val_mean_iou)
        loss_history.append(np.array(losses).mean())
        train_history.append(np.array(train_iou).mean())
        val_history.append(val_mean_iou)
        
        print('Epoch : {}/{}'.format(epoch+1, num_epochs))
        print('loss: {:.3f} - dice_coef: {:.3f} - val_dice_coef: {:.3f} - current lr: {}'.format(np.array(losses).mean(),
                                                                               np.array(train_iou).mean(),
                                                                               val_mean_iou,
                                                                                optimizer.param_groups[0]['lr']))
        
        if val_mean_iou > best_val_score:
            print(f'New score: {val_mean_iou:.3f}\tPrevious score: {best_val_score:.3f}')
            best_val_score = val_mean_iou
            best_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)
            
    model.load_state_dict(best_wts)
    print(f"Training Completed\nBest valid score: {best_val_score}")
    return loss_history, train_history, val_history