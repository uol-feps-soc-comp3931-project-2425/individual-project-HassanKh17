import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R

def compute_add(R_pred, t_pred, R_gt, t_gt, model_points):
    """
    Average Distance (ADD) metric - measures mean 3D point distance
    between predicted and ground truth poses.
    """
    # Transform model points with predicted and GT poses
    pred_pts = (R_pred @ model_points.T).T + t_pred
    gt_pts = (R_gt @ model_points.T).T + t_gt
    
    # Compute Euclidean distances
    return np.mean(np.linalg.norm(pred_pts - gt_pts, axis=1))

def compute_add_s(R_pred, t_pred, R_gt, t_gt, model_points):
    """
    Symmetric ADD (ADD-S) metric - uses closest point distances
    to handle symmetric objects.
    """
    pred_pts = (R_pred @ model_points.T).T + t_pred
    gt_pts = (R_gt @ model_points.T).T + t_gt
    
    # Find nearest neighbors
    tree = KDTree(gt_pts)
    distances, _ = tree.query(pred_pts)
    return np.mean(distances)

def compute_2d_projection_error(camera_matrix, R_pred, t_pred, R_gt, t_gt, model_points):
    """
    2D projection error - measures pixel distance between projected
    model points using predicted vs GT poses.
    """
    # Project points with predicted pose
    P_pred = camera_matrix @ np.hstack((R_pred, t_pred.reshape(3, 1)))
    points_h = np.hstack((model_points, np.ones((model_points.shape[0], 1)))).T
    pred_2d = (P_pred @ points_h).T
    pred_2d /= pred_2d[:, 2][:, None]  # Homogeneous division
    
    # Project points with GT pose
    P_gt = camera_matrix @ np.hstack((R_gt, t_gt.reshape(3, 1)))
    gt_2d = (P_gt @ points_h).T
    gt_2d /= gt_2d[:, 2][:, None]
    
    # Compute pixel distances
    return np.mean(np.linalg.norm(pred_2d[:, :2] - gt_2d[:, :2], axis=1))

def evaluate_predictions(preds, annotations, model_points, camera_matrix, obj_diameter):
    """
    Full evaluation pipeline computing ADD, ADD-S and 2D projection error.
    Prints metrics and accuracy under common thresholds.
    """
    pred_rotations, gt_rotations = [], []
    pred_translations, gt_translations = [], []
    
    # Convert all predictions and ground truth to rotation matrices
    for pred, ann in zip(preds, annotations):
        R_pred = R.from_quat(pred['rotation']).as_matrix()
        R_gt = R.from_quat(ann['rotation']).as_matrix()
        
        pred_rotations.append(R_pred)
        gt_rotations.append(R_gt)
        pred_translations.append(np.array(pred['translation']))
        gt_translations.append(np.array(ann['translation']))
    
    # Compute metrics for all samples
    add_scores = [compute_add(Rp, tp, Rg, tg, model_points)
                 for Rp, tp, Rg, tg in zip(pred_rotations, pred_translations, gt_rotations, gt_translations)]
    
    adds_scores = [compute_add_s(Rp, tp, Rg, tg, model_points)
                  for Rp, tp, Rg, tg in zip(pred_rotations, pred_translations, gt_rotations, gt_translations)]
    
    proj_errors = [compute_2d_projection_error(camera_matrix, Rp, tp, Rg, tg, model_points)
                  for Rp, tp, Rg, tg in zip(pred_rotations, pred_translations, gt_rotations, gt_translations)]
    
    # Print results
    print("=== Evaluation Results ===")
    print(f"ADD: {np.mean(add_scores):.2f} ± {np.std(add_scores):.2f} mm")
    print(f"ADD-S: {np.mean(adds_scores):.2f} ± {np.std(adds_scores):.2f} mm")
    print(f"2D Projection Error: {np.mean(proj_errors):.2f} ± {np.std(proj_errors):.2f} px")
    
    # Compute accuracy under thresholds
    def accuracy(metrics, threshold):
        return 100 * np.sum(np.array(metrics) < threshold) / len(metrics)
    
    print("\n=== Accuracy Under Thresholds ===")
    print(f"ADD < 10% diameter: {accuracy(add_scores, 0.1 * obj_diameter):.2f}%")
    print(f"ADD-S < 10% diameter: {accuracy(adds_scores, 0.1 * obj_diameter):.2f}%")
    print(f"2D Projection Error < 5px: {accuracy(proj_errors, 5.0):.2f}%")
    
    return {
        'add': add_scores,
        'add_s': adds_scores,
        'proj_error': proj_errors
    }