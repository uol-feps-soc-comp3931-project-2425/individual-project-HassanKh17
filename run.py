import torch
from model.posecnn import PoseCNN
from data.dataloader import create_dataloader
from training.train import train_model, plot_losses
from training.metrics import evaluate_predictions
from visualisation.draw_pose import draw_pose_axes, extract_mask_keypoints
from utils.output_manager import OutputManager
import json
import numpy as np

def visualise_predictions(model, dataloader, output_manager, camera_matrix, num_samples=5):
    """Visualise predictions on sample images"""
    model.eval()
    
    for i, (images, gt_trans, gt_rot) in enumerate(dataloader):
        if i >= num_samples:  # Only visualize first N samples
            break
            
        # Get prediction
        with torch.no_grad():
            pred_trans, pred_rot = model(images.to('cuda'))
            pred_trans = pred_trans.cpu().numpy()[0] * 1000  # Convert to mm
            pred_rot = pred_rot.cpu().numpy()[0]
            pred_rot = pred_rot / np.linalg.norm(pred_rot)  # Normalise
            
        # Convert image tensor to numpy
        img_np = images[0].permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8)
        
        # Get mask if available ( 4-channel input: RGB + mask)
        mask = None
        if images.shape[1] == 4:  # RGB + mask
            mask = (images[0][3].cpu().numpy() * 255).astype(np.uint8)
            origin = extract_mask_keypoints(mask)["center"] if extract_mask_keypoints(mask) else None
        else:
            origin = None
        
        # Draw predicted pose
        vis_img = draw_pose_axes(
            image=img_np.copy(),
            translation=pred_trans,
            quaternion=pred_rot,
            camera_matrix=camera_matrix,
            origin_2d=origin
        )
        
        # Save visualisation
        output_manager.save_visualisation(vis_img, f"pred_{i}")
        

def main():
    # Configuration
    config = {
        'data_root': "MFB_TRAIN/TRAIN",
        'batch_size': 16,
        'epochs': 20,
        'lr': 0.001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    output_manager = OutputManager(base_path="outputs", experiment_name="posecnn_experiment")
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
    
    output_manager.save_loss_plot(train_losses)
    
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
        obj_diameter=150.0  #
    )
    output_manager.save_evaluation_results(results)
    # Visualise predictions
    visualise_predictions(
        model=model,
        dataloader=train_loader,
        output_dir=config['output_dir'],
        camera_matrix=camera_matrix,
        num_samples=5  # Number of samples to visualise
    )
    print("\nTraining complete!")

if __name__ == '__main__':
    main()
