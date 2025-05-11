import os
import json
import torch
import numpy as np
from datetime import datetime
from typing import Dict, Any
import matplotlib.pyplot as plt
import cv2

class OutputManager:
    """
    Handles saving and organizing all training outputs including:
    - Model checkpoints
    - Training logs
    - Visualizations
    - Evaluation results
    """
    
    def __init__(self, base_path: str = "outputs", experiment_name: str = None):
        """
        Args:
            base_path: Root directory for all outputs
            experiment_name: Optional name for current experiment (uses timestamp if None)
        """
        # Create experiment directory
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_path = os.path.join(base_path, experiment_name)
        
        # Create subdirectories
        self.checkpoint_dir = os.path.join(self.experiment_path, "checkpoints")
        self.log_dir = os.path.join(self.experiment_path, "logs")
        self.vis_dir = os.path.join(self.experiment_path, "visualizations")
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.vis_dir, exist_ok=True)
        
        # Initialize log file
        self.log_file = os.path.join(self.log_dir, "training_log.jsonl")
        self._init_log_file()
        
    def _init_log_file(self):
        """Initialize the log file with header if needed"""
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("")  # Just create empty file
    
    def save_checkpoint(self, 
                      model: torch.nn.Module, 
                      epoch: int, 
                      optimizer: torch.optim.Optimizer = None,
                      metrics: Dict[str, Any] = None,
                      filename: str = None):
        """
        Save model checkpoint with additional training state
        
        Args:
            model: Model to save
            epoch: Current epoch number
            optimizer: Optimizer state 
            metrics: Dictionary of metrics to save 
            filename: Custom filename 
        """
        if filename is None:
            filename = f"checkpoint_epoch{epoch:03d}.pth"
            
        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'metrics': metrics
        }
        
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path
    
    def log_metrics(self, 
                   epoch: int, 
                   train_metrics: Dict[str, float], 
                   val_metrics: Dict[str, float] = None):
        """
        Log training and validation metrics to JSONL file
        
        Args:
            epoch: Current epoch number
            train_metrics: Dictionary of training metrics
            val_metrics: Dictionary of validation metrics
        """
        log_entry = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'train': train_metrics,
            'val': val_metrics
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def save_visualisation(self, 
                         image: np.ndarray, 
                         name: str, 
                         epoch: int = None,
                         convert_color=True):
        """
        Save visualisation image to visualisations directory
        
        Args:
            image: Image array to save
            name: Base name for the file
            epoch: Optional epoch number to include in filename
        """
        if epoch is not None:
            name = f"epoch{epoch:03d}_{name}"
            
        vis_path = os.path.join(self.vis_dir, f"{name}.png")
        # Save the image to the specified path using OpenCV
        cv2.imwrite(vis_path, image)
        return vis_path
    
    def save_loss_plot(self, 
                      train_losses: list, 
                      val_losses: list = None,
                      filename: str = "loss_curve.png"):
        """
        Save loss curve plot to visualisations directory
        
        Args:
            train_losses: List of training losses per epoch
            val_losses: List of validation losses per epoch
            filename: Name for the plot file
        """
        plt.figure()
        plt.plot(train_losses, label='Training Loss')
        if val_losses:
            plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True)
        
        plot_path = os.path.join(self.vis_dir, filename)
        plt.savefig(plot_path)
        plt.close()
        return plot_path
    
    def save_evaluation_results(self, 
                              metrics: Dict[str, Any], 
                              filename: str = "evaluation_results.json"):
        """
        Save evaluation metrics to JSON file
        
        Args:
            metrics: Dictionary of evaluation metrics
            filename: Name for the results file
        """
        results_path = os.path.join(self.log_dir, filename)
        with open(results_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        return results_path
