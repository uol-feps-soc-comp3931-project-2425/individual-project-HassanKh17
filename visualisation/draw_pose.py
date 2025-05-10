# pose_estimation_project/visualization/draw_pose.py
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

def draw_pose_axes(image, translation, quaternion, camera_matrix, axis_length=40, origin=None):
    """
    Draw 3D coordinate axes on image based on predicted pose.
    
    Args:
        image: Input RGB image (numpy array)
        translation: Predicted translation vector (3D)
        quaternion: Predicted rotation quaternion (4D)
        camera_matrix: Camera intrinsic matrix (3x3)
        axis_length: Length of axes in 3D space
        origin: Optional 2D point to draw axes from (if None, uses projection)
    """
    # Define 3D axis points
    axis = np.float32([
        [axis_length, 0, 0],  # X-axis
        [0, axis_length, 0],   # Y-axis
        [0, 0, axis_length]    # Z-axis
    ]).reshape(-1, 3)
    
    # Convert quaternion to rotation matrix
    rot_matrix = R.from_quat(quaternion).as_matrix()
    rvec, _ = cv2.Rodrigues(rot_matrix)
    tvec = translation.reshape(3, 1)
    
    # Project 3D axes to 2D image points
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, None)
    imgpts = np.int32(imgpts).reshape(-1, 2)
    
    # Get origin point (either projected or specified)
    if origin is None:
        origin, _ = cv2.projectPoints(np.zeros((1, 3)), rvec, tvec, camera_matrix, None)
        origin = tuple(np.int32(origin).reshape(2))
    else:
        origin = tuple(origin)
    
    # Draw colored axes
    image = cv2.line(image, origin, tuple(imgpts[0]), (0, 0, 255), 3)  # X (red)
    image = cv2.line(image, origin, tuple(imgpts[1]), (0, 255, 0), 3)  # Y (green)
    image = cv2.line(image, origin, tuple(imgpts[2]), (255, 0, 0), 3)  # Z (blue)
    
    return image

def extract_mask_keypoints(mask):
    """
    Extract keypoints from binary mask (centroid and extremities)
    Returns dict with 'center', 'left', 'right', 'top', 'bottom' points
    """
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get largest contour
    largest = max(contours, key=cv2.contourArea)
    
    # Calculate moments for centroid
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
        
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # Get extremities
    keypoints = {
        "left": tuple(largest[largest[:, :, 0].argmin()][0]),
        "right": tuple(largest[largest[:, :, 0].argmax()][0]),
        "top": tuple(largest[largest[:, :, 1].argmin()][0]),
        "bottom": tuple(largest[largest[:, :, 1].argmax()][0]),
        "center": (cx, cy)
    }
    
    return keypoints