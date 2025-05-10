import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from model.loss_functions import PoseLoss
from utils.output_manager import OutputManager

def train_model(model, train_loader, val_loader=None, epochs=20, lr=0.001, device='cuda', experiment_name='posecnn_experiment'):
    """
    Training loop for PoseCNN model
    Returns:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch (if val_loader provided)
    """
    #Initialize output manager
    output=OutputManager(experiment_name=experiment_name)

    # Setup optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # Halve LR every 10 epochs
    
    # Loss function
    criterion = PoseLoss()
    
    train_losses = []
    val_losses = []
    
    # Move model to device
    model = model.to(device)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        # Training loop with progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for images, trans, rot in pbar:
            # Move data to device
            images = images.to(device)
            trans = trans.to(device)
            rot = rot.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            pred_trans, pred_rot = model(images)
            
            # Compute loss
            loss, loss_dict = criterion(pred_trans, pred_rot, trans, rot)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        # Average loss for epoch
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Validation if val_loader provided
        if val_loader:
            val_loss = validate_model(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            print(f'Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}')
        else:
            print(f'Epoch {epoch+1} - Train Loss: {avg_loss:.4f}')
        
        # Step scheduler
        scheduler.step()
    output.save_loss_plot(train_losses, val_losses)    
    
    return train_losses, val_losses

def validate_model(model, val_loader, criterion, device):
    """Validation loop"""
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for images, trans, rot in val_loader:
            images = images.to(device)
            trans = trans.to(device)
            rot = rot.to(device)
            
            pred_trans, pred_rot = model(images)
            loss, _ = criterion(pred_trans, pred_rot, trans, rot)
            val_loss += loss.item()
    
    return val_loss / len(val_loader)

def save_checkpoint(model, epoch, loss, path='checkpoints'):
    """Save model checkpoint"""
    os.makedirs(path, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }, f'{path}/checkpoint_epoch{epoch}.pth')

def plot_losses(train_losses, val_losses=None, save_path='loss_plot.png'):
    """Plot training and validation loss curves"""
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    if val_losses:
        plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()