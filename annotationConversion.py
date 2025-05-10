import numpy as np
import os
import json
from scipy.spatial.transform import Rotation as R

pose_folder = "MFB_TRAIN/TRAIN/pose"
output_json = "MFB_TRAIN/posecnn_annotations.json"

posecnn_annotations = []

for pose_file in os.listdir(pose_folder):
    if pose_file.endswith(".npy"):
        # Load the .npy file
        pose_matrix = np.load(os.path.join(pose_folder, pose_file))

        # Extract translation (last column of the matrix)
        translation = pose_matrix[:, 3].tolist()

        # Extract rotation matrix (first 3x3 block)
        rotation_matrix = pose_matrix[:, :3]

        # Convert rotation matrix to quaternion
        quaternion = R.from_matrix(rotation_matrix).as_quat().tolist()

        # Extract image ID from filename (assuming "1.npy" → image_id = 1)
        image_id = int(os.path.splitext(pose_file)[0])

        # Store in PoseCNN format
        posecnn_annotations.append({
            "image_id": image_id,
            "object_class": 0,  # Update with actual class if needed
            "translation": translation,
            "rotation": quaternion
        })

# Save the converted data
with open(output_json, "w") as f:
    json.dump(posecnn_annotations, f)

print("All .npy files converted successfully to PoseCNN format!")
