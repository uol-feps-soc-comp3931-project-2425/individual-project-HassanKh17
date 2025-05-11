import torch
from model.posecnn import PoseCNN
from data.dataloader import create_dataloader
from training.train import train_model, plot_losses
from training.metrics import evaluate_predictions
from visualisation.draw_pose import draw_pose_axes, extract_mask_keypoints
from utils.output_manager import OutputManager
import json
import numpy as np
import cv2
import torchvision.transforms.functional as TF
import os

def visualise_predictions(model, annotations, image_dir, mask_dir, output_manager, camera_matrix, device, num_samples=5):
    """Visualise predictions on sample images"""
    model.eval()  # Set the model to evaluation mode
    
    for i, ann in enumerate(annotations[:num_samples]):  # Iterate over a subset of annotations
        img_id = ann['image_id']  # Get the image ID
        img_path = os.path.join(image_dir, f"{img_id}.png")  # Construct the image path
        mask_path = os.path.join(mask_dir, f"{img_id}.png")  # Construct the mask path

        image = cv2.imread(img_path)  # Read the image
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Read the mask in grayscale

        if image is None or mask is None:  # Skip if image or mask is missing
            continue

        # Resize and preprocess the image and mask
        image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (640, 480))
        mask = cv2.resize(mask, (640, 480)) / 255.0
        stacked = np.dstack((image, mask)).astype(np.float32)  # Stack image and mask
        input_tensor = TF.to_tensor(stacked).unsqueeze(0).to(device)  # Convert to tensor and move to device

        with torch.no_grad():  # Disable gradient computation for inference
            pred_trans, pred_rot = model(input_tensor)  # Get predictions for translation and rotation
            trans = pred_trans[0].cpu().numpy() * 1000  # Convert translation to mm
            quat = pred_rot[0].cpu().numpy()  # Get quaternion
            quat = quat / np.linalg.norm(quat) if np.linalg.norm(quat) > 0 else np.array([0, 0, 0, 1])  # Normalize quaternion

        # Extract keypoints from the mask
        mask_uint8 = (mask * 255).astype(np.uint8)  # Convert mask to uint8
        keypoints = extract_mask_keypoints(mask_uint8)  # Extract keypoints
        if keypoints is None:  # Skip if no keypoints are found
            continue

        axis_origin = keypoints["center"]  # Use the center keypoint as the origin for pose axes

        # Draw pose axes on the image
        vis_img = draw_pose_axes(image.copy(), trans, quat, camera_matrix, origin=axis_origin)
        if vis_img.dtype != np.uint8:  # Ensure the image is in uint8 format
            vis_img = np.clip(vis_img, 0, 255).astype(np.uint8)
        if vis_img.ndim == 2 or vis_img.shape[2] == 1:  # Convert grayscale to RGB if needed
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2RGB)

        # Draw keypoints on the image
        color_map = {  # Define colors for each keypoint label
            "left": (0, 255, 255),
            "right": (255, 255, 0),
            "top": (255, 0, 255),
            "bottom": (0, 128, 255),
            "center": (255, 0, 0)
        }
        for label, pt in keypoints.items():  # Iterate over keypoints
            cv2.circle(vis_img, pt, 5, color_map[label], -1)  # Draw a circle for each keypoint
            cv2.putText(vis_img, label, (pt[0] + 5, pt[1] - 5),  # Add a label near the keypoint
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[label], 1)

        # Save the visualisation using OutputManager
        output_manager.save_visualisation(cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR), f"vis_{img_id}")

def main():
    # Configuration
    config = {
        'data_root': "MFB_TRAIN/TRAIN",  # Root directory for data
        'batch_size': 16,  # Batch size for training
        'epochs': 20,  # Number of training epochs
        'lr': 0.001,  # Learning rate
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
    }
    device = config['device']  # Set the device
    output_manager = OutputManager(base_path="outputs", experiment_name="posecnn_experiment")  # Initialize OutputManager

    # Create model
    model = PoseCNN()  # Instantiate the PoseCNN model
    
    # Create data loaders
    image_dir = f"{config['data_root']}/image_resized"  # Directory for resized images
    annotation_file = f"{config['data_root']}/posecnn_annotations.json"  # Path to annotation file
    mask_dir = f"{config['data_root']}/mask"  # Directory for masks
    
    train_loader = create_dataloader(  # Create the training data loader
        image_dir,
        annotation_file,
        mask_dir,
        batch_size=config['batch_size']
    )
    
    # Train model
    print("Starting training...")
    train_losses, _ = train_model(  # Train the model
        model,
        train_loader,
        epochs=config['epochs'],
        lr=config['lr'],
        device=config['device']
    )
    
    output_manager.save_loss_plot(train_losses)  # Save the training loss plot
    
    # Evaluation
    print("\nEvaluating model...")
    with open(annotation_file) as f:  # Load annotations
        annotations = json.load(f)
    
    # Load model points and camera matrix
    model_points = np.load(f"{config['data_root']}/model_keypoints.npy")  # Load model keypoints
    camera_matrix = np.load(f"{config['data_root']}/camera_matrix.npy")  # Load camera matrix
    
    # Generate predictions (in practice would use separate test set)
    predictions = []
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation for inference
        for images, _, _ in train_loader:  # Iterate over training data
            images = images.to(config['device'])  # Move images to device
            pred_trans, pred_rot = model(images)  # Get predictions
            
            # Convert predictions to numpy and scale translation back to mm
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
    
    # Evaluate predictions
    results = evaluate_predictions(
        predictions,
        annotations,
        model_points,
        camera_matrix,
        obj_diameter=150.0  # Object diameter for evaluation
    )
    output_manager.save_evaluation_results(results)  # Save evaluation results
    
    # Visualise predictions
    visualise_predictions(
        model=model,
        annotations=annotations,
        image_dir=image_dir,
        mask_dir=mask_dir,
        output_manager=output_manager,
        camera_matrix=camera_matrix,
        device=device,
        num_samples=5  # Number of samples to visualise
    )
    print("\nTraining complete!")

if __name__ == '__main__':
    main()  # Run the main function
