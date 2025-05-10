import torch
from model.posecnn import PoseCNN
from data.dataloader import create_dataloader
from training.train import train_model, plot_losses
from training.metrics import evaluate_predictions
import json
import numpy as np

def main():
    # Configuration
    config = {
        'data_root': r"C:\Users\lunap\Downloads\MFB_TRAIN\TRAIN",
        'batch_size': 16,
        'epochs': 20,
        'lr': 0.001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # Create model
    model = PoseCNN()
    
    # Create data loaders
    image_dir = f"{config['data_root']}/image_resized"
    annotation_file = f"{config['data_root']}/posecnn_annotations.json"
    mask_dir = f"{config['data_root']}/mask"
    
    train_loader = create_dataloader(
        image_dir,
        annotation_file,
        mask_dir,
        batch_size=config['batch_size']
    )
    
    # Train model
    print("Starting training...")
    train_losses, _ = train_model(
        model,
        train_loader,
        epochs=config['epochs'],
        lr=config['lr'],
        device=config['device']
    )
    
    # Save loss plot
    plot_losses(train_losses, save_path='training_loss.png')
    
    # Evaluation
    print("\nEvaluating model...")
    with open(annotation_file) as f:
        annotations = json.load(f)
    
    # Load model points and camera matrix
    model_points = np.load(f"{config['data_root']}/model_keypoints.npy")
    camera_matrix = np.load(f"{config['data_root']}/camera_matrix.npy")
    
    # Generate predictions (in practice would use separate test set)
    predictions = []
    model.eval()
    with torch.no_grad():
        for images, _, _ in train_loader:
            images = images.to(config['device'])
            pred_trans, pred_rot = model(images)
            
            # Convert to numpy and scale translation back to mm
            pred_trans = pred_trans.cpu().numpy() * 1000
            pred_rot = pred_rot.cpu().numpy()
            
            # Normalize quaternions
            norms = np.linalg.norm(pred_rot, axis=1, keepdims=True)
            pred_rot = np.where(norms > 0, pred_rot / norms, [0, 0, 0, 1])
            
            # Store predictions
            for i in range(len(pred_trans)):
                predictions.append({
                    'translation': pred_trans[i].tolist(),
                    'rotation': pred_rot[i].tolist()
                })
    
    # Evaluate
    results = evaluate_predictions(
        predictions,
        annotations,
        model_points,
        camera_matrix,
        obj_diameter=150.0  # Adjust based on your object
    )
    
    print("\nTraining complete!")

if __name__ == '__main__':
    main()