# pose_estimation_project/model/loss_functions.py
import torch
import torch.nn as nn

def quaternion_cosine_loss(q_pred, q_true):
    """
    Loss function for quaternion prediction.
    Measures angular difference between predicted and ground truth quaternions.
    """
    # Normalize both quaternions
    q_pred = q_pred / torch.norm(q_pred, dim=1, keepdim=True)
    q_true = q_true / torch.norm(q_true, dim=1, keepdim=True)
    
    # Cosine similarity (clamped for numerical stability)
    cos_sim = torch.sum(q_pred * q_true, dim=1).clamp(-1 + 1e-6, 1 - 1e-6)
    
    # Convert to angular difference
    return 1 - torch.mean(cos_sim)

class PoseLoss(nn.Module):
    """Combined loss for translation and rotation"""
    def __init__(self):
        super().__init__()
        self.trans_loss = nn.SmoothL1Loss()  # For translation
        
    def forward(self, pred_trans, pred_rot, gt_trans, gt_rot):
        # Translation loss (L1 smooth)
        trans_loss = self.trans_loss(pred_trans, gt_trans)
        
        # Rotation loss (quaternion cosine)
        rot_loss = quaternion_cosine_loss(pred_rot, gt_rot)
        
        # Combined loss
        total_loss = trans_loss + rot_loss
        return total_loss, {'trans_loss': trans_loss, 'rot_loss': rot_loss}