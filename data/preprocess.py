import os
import cv2
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

def preprocess_dataset(dataset_root):
    """Run one-time preprocessing for resizing images and converting poses"""
    # Define paths
    dataset_root = os.path.normpath(dataset_root)
    if not os.path.isdir(dataset_root):
        raise NotADirectoryError(f"Dataset root not found: {dataset_root}")
    image_dir = os.path.join(dataset_root, "image")
    pose_dir = os.path.join(dataset_root, "pose")
    resized_dir = os.path.join(dataset_root, "image_resized")
    annotation_path = os.path.join(dataset_root, "posecnn_annotations.json")
    
   # Verify input directories exist
    if not os.path.isdir(image_dir):
        raise NotADirectoryError(f"Image directory not found: {image_dir}")
    if not os.path.isdir(pose_dir):
        raise NotADirectoryError(f"Pose directory not found: {pose_dir}")

    # Only run preprocessing if outputs don't exist
    needs_resize = not os.path.isdir(resized_dir)
    needs_annotations = not os.path.exists(annotation_path)

    if needs_resize or needs_annotations:
        print("🚀 Starting dataset preprocessing...")
        if needs_resize:
            _resize_images(image_dir, resized_dir)
        if needs_annotations:
            _generate_annotations(pose_dir, annotation_path)
        print("✅ Preprocessing complete!")
    else:
        print("✅ Using existing preprocessed files")

def _resize_images(image_dir, resized_dir):
    """Resize images to 640x480 and save to new directory"""
    os.makedirs(resized_dir, exist_ok=True)
    
    print(f"🔄 Resizing images from {image_dir}...")
    for file in os.listdir(image_dir):
        if file.lower().endswith((".png", ".jpg")):
            img_path = os.path.join(image_dir, file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Skipped: {file}")
                continue
            resized = cv2.resize(img, (640, 480))
            cv2.imwrite(os.path.join(resized_dir, file), resized)
    
    print(f"✅ Resized images saved to: {resized_dir}")

def _generate_annotations(pose_dir, annotation_path):
    """Convert .npy pose files to JSON annotations"""
    annotations = []
    print(f"🔄 Generating pose annotations from {pose_dir}...")
    
    for file in os.listdir(pose_dir):
        if file.endswith(".npy"):
            pose_matrix = np.load(os.path.join(pose_dir, file))
            
            # Extract translation and rotation
            translation = pose_matrix[:, 3].tolist()
            rotation_matrix = pose_matrix[:, :3]
            quaternion = R.from_matrix(rotation_matrix).as_quat().tolist()
            
            # Get image ID from filename
            image_id = int(os.path.splitext(file)[0])
            
            annotations.append({
                "image_id": image_id,
                "object_class": 0,
                "translation": translation,
                "rotation": quaternion
            })
    
    # Save annotations
    with open(annotation_path, "w") as f:
        json.dump(annotations, f, indent=2)
    
    print(f"✅ Saved {len(annotations)} pose annotations to: {annotation_path}")
