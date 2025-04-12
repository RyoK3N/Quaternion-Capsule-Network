import os 
import json 
import logging 
import random 
import numpy as np 
import tensorflow as tf 
from tqdm import tqdm 
from datetime import datetime 
import warnings 
import argparse
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

warnings.filterwarnings("ignore")

from QuaternionCapsules import  QuaternionCapsNet
from qdataloader import QCNDataloader
from loss import qrotation_loss, qtranslation_loss, qgeometric_loss, qreprojection_loss
from utils import plot_figs_to_png, get_parent_map

# -----------------------------------------------------------------------------
# Logging Runtime for naming logs/runs_{datetime}
# -----------------------------------------------------------------------------
name = datetime.now().strftime("%Y%m%d_%H%M%S")

# -----------------------------------------------------------------------------
# Directory Setup
# -----------------------------------------------------------------------------
BASE_LOG_DIR: str = os.path.join("logs", f"runs_{name}")
CHECKPOINT_DIR: str = os.path.join(BASE_LOG_DIR, "checkpoints")
IMAGES_DIR: str = os.path.join(BASE_LOG_DIR, "images")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Logging Setup
# -----------------------------------------------------------------------------
log_file: str = os.path.join(BASE_LOG_DIR, "train.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(log_file)]
)
logger: logging.Logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Load Configuration
# -----------------------------------------------------------------------------
config_path = './config_train.json'
try:
    with open(config_path, 'r') as f:
        config: dict = json.load(f)
except FileNotFoundError:
    logger.warning(f"Config file {config_path} not found. Using default configuration.")
    # Default configuration
    config = {
        "model": {
            "in_channels": 4,
            "out_caps": 1
        },
        "training": {
            "batch_size": 32,
            "epochs": 100,
            "learning_rate": 0.001
        },
        "loss_weights": {
            "rotation": 1.0,
            "translation": 0.1,
            "reprojection": 1.0,
            "geometric_alpha": 1.0,
            "geometric_beta": 0.1,
            "geometric_gamma": 0.1,
            "unit_constraint": 0.1
        },
        "routing": {
            "method": "dynamic",
            "iterations": 3
        },
        "visualization": { # Default visualization config
            "plot_every_n_epochs": 1,
            "plot_sample_index": 0,
            "plot_save_subdir": "epoch_plots",
            "num_plot_samples": 3
        }
    }

# Get loss weights from config
loss_weights = config.get("loss_weights", {})
ALPHA = loss_weights.get("geometric_alpha", 1.0)
BETA = loss_weights.get("geometric_beta", 0.1)
GAMMA = loss_weights.get("geometric_gamma", 0.1)
UNIT_WEIGHT = loss_weights.get("unit_constraint", 0.1) # Get unit constraint weight

# Load visualization config
vis_config = config.get("visualization", {})
PLOT_EVERY = vis_config.get("plot_every_n_epochs", 1)
NUM_PLOT_SAMPLES = vis_config.get("num_plot_samples", 3) # Get number of samples
PLOT_SUBDIR = vis_config.get("plot_save_subdir", "epoch_plots")
# Extract viewport tuple/list
VIEWPORT_PARAMS = (
    vis_config.get("viewport", {}).get("x", 0),
    vis_config.get("viewport", {}).get("y", 0),
    vis_config.get("viewport", {}).get("width", 1920),
    vis_config.get("viewport", {}).get("height", 1080)
)

@tf.function
def train_step(model, optimizer, x_batch, y_rot, y_trans, y_kp2d_gt, y_kp3d_world, proj_matrix, viewport):
    """Single training step with gradient calculation and optimization"""
    with tf.GradientTape() as tape:
        # Forward pass
        y_pred = model(x_batch, training=True)
        
        # Split prediction
        pred_rot = y_pred[:, :4]
        pred_trans = y_pred[:, 4:7]
        
        # Calculate loss using qgeometric_loss
        loss = qgeometric_loss(y_kp2d_gt, y_kp3d_world, y_rot, y_trans, pred_rot, pred_trans, proj_matrix, viewport,
                               alpha=ALPHA, beta=BETA, gamma=GAMMA, unit_weight=UNIT_WEIGHT)
    
    # Calculate gradients and update model
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    # Calculate individual metrics for logging
    rot_metric = qrotation_loss(y_rot, pred_rot)
    trans_metric = qtranslation_loss(y_trans, pred_trans)
    reproj_metric = qreprojection_loss(y_kp2d_gt, y_kp3d_world, y_rot, y_trans, pred_rot, pred_trans, proj_matrix, viewport)
    
    return loss, rot_metric, trans_metric, reproj_metric

def val_step(model, x_batch, y_rot, y_trans, y_kp2d_gt, y_kp3d_world, proj_matrix, viewport):
    """Validation step without gradient calculation"""
    # Forward pass
    y_pred = model(x_batch, training=False)
    
    # Split prediction
    pred_rot = y_pred[:, :4]
    pred_trans = y_pred[:, 4:7]
    
    # Calculate loss using qgeometric_loss
    loss = qgeometric_loss(y_kp2d_gt, y_kp3d_world, y_rot, y_trans, pred_rot, pred_trans, proj_matrix, viewport,
                           alpha=ALPHA, beta=BETA, gamma=GAMMA, unit_weight=UNIT_WEIGHT)
    
    # Calculate metrics
    rot_metric = qrotation_loss(y_rot, pred_rot)
    trans_metric = qtranslation_loss(y_trans, pred_trans)
    reproj_metric = qreprojection_loss(y_kp2d_gt, y_kp3d_world, y_rot, y_trans, pred_rot, pred_trans, proj_matrix, viewport)
    
    return loss, rot_metric, trans_metric, reproj_metric

def evaluate_model(model, test_dataset):
    """Evaluate model on test dataset and return metrics"""
    test_loss = 0
    test_rot_loss = 0
    test_trans_loss = 0
    test_reproj_loss = 0
    num_test_batches = 0
    
    # Store predictions and ground truth for visualization
    all_pred_rots = []
    all_true_rots = []
    all_pred_trans = []
    all_true_trans = []
    
    for kp2d_quat, view_rot_quat, view_trans, kp2d_gt, kp3d_world, proj_matrix in test_dataset:
        # Get viewport params (assuming constant for eval set, could also be yielded by loader)
        viewport_params = VIEWPORT_PARAMS 
        batch_size = tf.shape(kp2d_quat)[0]
        x_input = tf.reshape(kp2d_quat, [batch_size, kp2d_quat.shape[1], 1, kp2d_quat.shape[2]])
        
        # Forward pass
        y_pred = model(x_input, training=False)
        
        # Split prediction
        pred_rot = y_pred[:, :4]
        pred_trans = y_pred[:, 4:7]
        
        # Calculate overall loss using qgeometric_loss for consistency
        loss = qgeometric_loss(kp2d_gt, kp3d_world, view_rot_quat, view_trans, pred_rot, pred_trans, proj_matrix, viewport_params,
                                alpha=ALPHA, beta=BETA, gamma=GAMMA, unit_weight=UNIT_WEIGHT)

        # Calculate individual component losses for detailed reporting
        rot_loss = qrotation_loss(view_rot_quat, pred_rot)
        trans_loss = qtranslation_loss(view_trans, pred_trans)
        reproj_loss = qreprojection_loss(kp2d_gt, kp3d_world, view_rot_quat, view_trans, pred_rot, pred_trans, proj_matrix, viewport_params)
        
        test_loss += loss
        test_rot_loss += rot_loss
        test_trans_loss += trans_loss
        test_reproj_loss += reproj_loss
        num_test_batches += 1
        
        # Store for visualization
        all_pred_rots.append(pred_rot.numpy())
        all_true_rots.append(view_rot_quat.numpy())
        all_pred_trans.append(pred_trans.numpy())
        all_true_trans.append(view_trans.numpy())
    
    # Calculate average
    test_loss /= num_test_batches
    test_rot_loss /= num_test_batches
    test_trans_loss /= num_test_batches
    test_reproj_loss /= num_test_batches
    
    # Concatenate all batches
    all_pred_rots = np.concatenate(all_pred_rots, axis=0)
    all_true_rots = np.concatenate(all_true_rots, axis=0)
    all_pred_trans = np.concatenate(all_pred_trans, axis=0)
    all_true_trans = np.concatenate(all_true_trans, axis=0)
    
    return {
        'test_loss': test_loss.numpy(),
        'test_rot_loss': test_rot_loss.numpy(),
        'test_trans_loss': test_trans_loss.numpy(),
        'test_reproj_loss': test_reproj_loss.numpy(),
        'pred_rots': all_pred_rots,
        'true_rots': all_true_rots,
        'pred_trans': all_pred_trans,
        'true_trans': all_true_trans
    }

def visualize_results(results, save_dir):
    """Visualize and save evaluation results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Translation error histogram
    trans_errors = np.sqrt(np.sum((results['pred_trans'] - results['true_trans'])**2, axis=1))
    plt.figure(figsize=(10, 6))
    plt.hist(trans_errors, bins=50)
    plt.title('Translation Error Distribution')
    plt.xlabel('Euclidean Distance Error')
    plt.ylabel('Count')
    plt.savefig(os.path.join(save_dir, 'translation_error_hist.png'))
    plt.close()
    
    # Rotation error histogram
    rot_errors = []
    for i in range(len(results['pred_rots'])):
        # Convert quaternions from [w, x, y, z] to [x, y, z, w] format for scipy
        q1_wxyz = results['pred_rots'][i]
        q2_wxyz = results['true_rots'][i]
        
        q1_xyzw = np.array([q1_wxyz[1], q1_wxyz[2], q1_wxyz[3], q1_wxyz[0]])
        q2_xyzw = np.array([q2_wxyz[1], q2_wxyz[2], q2_wxyz[3], q2_wxyz[0]])
        
        r1 = R.from_quat(q1_xyzw)
        r2 = R.from_quat(q2_xyzw)
        
        # Calculate angular difference in degrees
        relative_rot = r1.inv() * r2
        angle_diff = np.degrees(np.abs(relative_rot.magnitude()))
        rot_errors.append(angle_diff)
    
    plt.figure(figsize=(10, 6))
    plt.hist(rot_errors, bins=50)
    plt.title('Rotation Error Distribution')
    plt.xlabel('Angular Error (degrees)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(save_dir, 'rotation_error_hist.png'))
    plt.close()
    
    # Translation components comparison
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    for i, axis_name in enumerate(['X', 'Y', 'Z']):
        axes[i].scatter(results['true_trans'][:, i], results['pred_trans'][:, i], alpha=0.5)
        axes[i].set_title(f'True vs Predicted Translation - {axis_name} Axis')
        axes[i].set_xlabel(f'True {axis_name}')
        axes[i].set_ylabel(f'Predicted {axis_name}')
        # Add diagonal line (perfect prediction)
        min_val = min(np.min(results['true_trans'][:, i]), np.min(results['pred_trans'][:, i]))
        max_val = max(np.max(results['true_trans'][:, i]), np.max(results['pred_trans'][:, i]))
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'translation_comparison.png'))
    plt.close()
    
    # Save summary statistics
    with open(os.path.join(save_dir, 'evaluation_summary.txt'), 'w') as f:
        f.write(f"Overall Test Loss (Geometric): {results['test_loss']:.4f}\n")
        f.write(f"Component - Rotation Loss: {results['test_rot_loss']:.4f}\n")
        f.write(f"Component - Translation Loss: {results['test_trans_loss']:.4f}\n")
        f.write(f"Component - Reprojection Loss: {results['test_reproj_loss']:.4f}\n")
        f.write(f"Mean Translation Error: {np.mean(trans_errors):.4f}\n")
        f.write(f"Median Translation Error: {np.median(trans_errors):.4f}\n")
        f.write(f"Mean Rotation Error (degrees): {np.mean(rot_errors):.4f}\n")
        f.write(f"Median Rotation Error (degrees): {np.median(rot_errors):.4f}\n")

# --- Visualization Helper Functions (Copied from viz.py) ---

def quaternion_and_translation_to_view_matrix(quat_wxyz, trans_xyz):
    """Converts wxyz quaternion and xyz translation to a 4x4 view matrix."""
    # Ensure quaternion is normalized
    quat_wxyz = quat_wxyz / (np.linalg.norm(quat_wxyz) + 1e-8)
    # Scipy expects [x, y, z, w]
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    rotation_matrix = R.from_quat(quat_xyzw).as_matrix()

    view_matrix = np.eye(4)
    view_matrix[:3, :3] = rotation_matrix
    view_matrix[:3, 3] = trans_xyz # Incorrect placement for OpenGL Style
    view_matrix[3, :3] = trans_xyz # Correct placement for OpenGL Style
    view_matrix[3, 3] = 1.0 # Ensure bottom-right is 1
    return view_matrix

def project_points(points_3d, view_matrix, proj_matrix, viewport):
    """Projects 3D points (N, 3) to 2D Screen Coords (N, 2) using matrices and viewport."""
    num_points = points_3d.shape[0]
    points_4d = np.hstack((points_3d, np.ones((num_points, 1))))
    # Try post-multiplying without transpose, assuming OpenGL style matrices
    points_view_h = points_4d @ view_matrix
    points_clip_h = points_view_h @ proj_matrix

    # Perspective divide
    w = points_clip_h[:, 3:4]
    valid_idx = np.abs(w) > 1e-6
    points_ndc = np.full_like(points_clip_h[:, :2], np.nan)
    w_safe = np.where(valid_idx, w, 1.0) # Avoid division by zero, put NaN later
    points_ndc = points_clip_h[:, :2] / w_safe
    points_ndc[~valid_idx.flatten()] = np.nan # Set invalid points to NaN

    # Apply Viewport Transform
    viewport_x, viewport_y, viewport_width, viewport_height = viewport
    x_ndc = points_ndc[:, 0]
    y_ndc = points_ndc[:, 1]
    x_screen = (x_ndc * 0.5 + 0.5) * viewport_width + viewport_x
    y_screen = (y_ndc * 0.5 + 0.5) * viewport_height + viewport_y # Assuming Y points down
    points_screen = np.stack([x_screen, y_screen], axis=-1)

    return points_screen

def plot_skeleton_3d(ax, points_3d, joint_order, parent_map, title, color='blue', limits=None):
    """Plots a 3D skeleton."""
    ax.clear()
    joint_to_idx = {name: i for i, name in enumerate(joint_order)}
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], color=color, s=30)
    for joint, parent in parent_map.items():
        if parent and joint in joint_to_idx and parent in joint_to_idx:
            idx_j = joint_to_idx[joint]
            idx_p = joint_to_idx[parent]
            ax.plot([points_3d[idx_p, 0], points_3d[idx_j, 0]],
                    [points_3d[idx_p, 1], points_3d[idx_j, 1]],
                    [points_3d[idx_p, 2], points_3d[idx_j, 2]],
                    color=color, linewidth=1.5)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_zlabel("Z")
    ax.set_title(title)
    if limits:
         ax.set_xlim(limits['xlim'])
         ax.set_ylim(limits['ylim'])
         ax.set_zlim(limits['zlim'])
    else:
        # Auto-scaling logic
        max_range = np.array([points_3d[:, 0].max()-points_3d[:, 0].min(),
                              points_3d[:, 1].max()-points_3d[:, 1].min(),
                              points_3d[:, 2].max()-points_3d[:, 2].min()]).max() / 2.0
        mid_x = (points_3d[:, 0].max()+points_3d[:, 0].min()) * 0.5
        mid_y = (points_3d[:, 1].max()+points_3d[:, 1].min()) * 0.5
        mid_z = (points_3d[:, 2].max()+points_3d[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.view_init(elev=10., azim=-60)
    ax.set_aspect('auto')

def plot_skeleton_2d(ax, points_2d, joint_order, parent_map, title, color='red', limits=None):
    """Plots a 2D skeleton."""
    ax.clear()
    joint_to_idx = {name: i for i, name in enumerate(joint_order)}
    valid_pts = ~np.isnan(points_2d).any(axis=1)
    ax.scatter(points_2d[valid_pts, 0], points_2d[valid_pts, 1], color=color, s=30)
    for joint, parent in parent_map.items():
        if parent and joint in joint_to_idx and parent in joint_to_idx:
            idx_j = joint_to_idx[joint]
            idx_p = joint_to_idx[parent]
            if valid_pts[idx_j] and valid_pts[idx_p]:
                ax.plot([points_2d[idx_p, 0], points_2d[idx_j, 0]],
                        [points_2d[idx_p, 1], points_2d[idx_j, 1]],
                        color=color, linewidth=1.5)
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title(title)
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    if limits:
        ax.set_xlim(limits['xlim'])
        ax.set_ylim(limits['ylim'])

# --- New Function for Interpolated Plotting --- 

def plot_skeleton_2d_interpolated(ax, points_2d, joint_order, parent_map, title, color='blue', limits_to_check=None):
    """Plots a 2D skeleton, showing all points and interpolating outside bounds."""
    ax.clear()
    joint_to_idx = {name: i for i, name in enumerate(joint_order)}
    num_points = points_2d.shape[0]
    valid_pts = ~np.isnan(points_2d).any(axis=1)
    
    points_inside = np.zeros(num_points, dtype=bool)
    if limits_to_check:
        x_lim = limits_to_check['xlim']
        y_lim = limits_to_check['ylim']
        points_inside = valid_pts & \
                        (points_2d[:, 0] >= x_lim[0]) & (points_2d[:, 0] <= x_lim[1]) & \
                        (points_2d[:, 1] >= y_lim[0]) & (points_2d[:, 1] <= y_lim[1])
    else:
        points_inside = valid_pts # If no limits, all valid points are considered 'inside'

    # Plot points (e.g., different markers/alpha for inside/outside)
    ax.scatter(points_2d[points_inside, 0], points_2d[points_inside, 1], color=color, s=30, label='Inside Viewport')
    ax.scatter(points_2d[valid_pts & ~points_inside, 0], points_2d[valid_pts & ~points_inside, 1], 
               color=color, s=20, alpha=0.5, marker='x', label='Outside Viewport')

    # Plot bones
    for joint, parent in parent_map.items():
        if parent and joint in joint_to_idx and parent in joint_to_idx:
            idx_j = joint_to_idx[joint]
            idx_p = joint_to_idx[parent]
            # Only plot bone if both points are numerically valid
            if valid_pts[idx_j] and valid_pts[idx_p]:
                # Check if both endpoints are inside the original limits
                is_inside = points_inside[idx_j] and points_inside[idx_p]
                linestyle = '-' if is_inside else ':'
                line_alpha = 1.0 if is_inside else 0.6
                ax.plot([points_2d[idx_p, 0], points_2d[idx_j, 0]],
                        [points_2d[idx_p, 1], points_2d[idx_j, 1]],
                        color=color, linewidth=1.5, linestyle=linestyle, alpha=line_alpha)

    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.set_title(title)
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    # Do not set limits here, let it auto-scale
    ax.legend(fontsize='small')

# --- Plotting Function for Training --- 

def generate_epoch_visualization(model, dataloader, config, epoch, save_dir, sample_indices):
    """Generates and saves visualization plots for specific sample indices."""
    # Get config details inside the loop for each sample
    vis_cfg = config.get("visualization", {})
    plot_subdir = vis_cfg.get("plot_save_subdir", "epoch_plots")
    viewport_params = (
        vis_cfg.get("viewport", {}).get("x", 0),
        vis_cfg.get("viewport", {}).get("y", 0),
        vis_cfg.get("viewport", {}).get("width", 1920),
        vis_cfg.get("viewport", {}).get("height", 1080)
    )
    proj_matrix = dataloader.projection_matrix
    parent_map = get_parent_map()
    joint_order = dataloader.joint_order

    # Create base save directory if it doesn't exist
    plot_base_dir = os.path.join(save_dir, plot_subdir)
    os.makedirs(plot_base_dir, exist_ok=True)

    logger.info(f"Generating visualizations for epoch {epoch+1} for indices: {sample_indices}")

    for sample_idx in sample_indices:
        try:
            # Get the specific sample using dataloader's __getitem__
            sample = dataloader.__getitem__(sample_idx)
            kp3d_world_gt = sample['kp3d_world']
            kp2d_cam_gt = sample['kp2d_gt'] # Ground truth 2D from dataset
            quat_gt = sample['view_rot_quat']
            trans_gt = sample['view_trans']
            kp2d_quat_input = sample['kp2d_quat'] # Model input

            # --- Get Model Prediction --- 
            # Prepare model input (add batch dimension and reshape)
            # Input shape: (1, num_joints, 1, 4)
            model_input = tf.reshape(kp2d_quat_input, [1, len(joint_order), 1, 4])
            prediction = model(model_input, training=False) # Shape: (1, 7)
            pred_rot = prediction[0, :4].numpy() # Predicted rotation quaternion
            pred_trans = prediction[0, 4:].numpy() # Predicted translation

            # --- Reconstruct Transforms --- 
            view_matrix_gt = quaternion_and_translation_to_view_matrix(quat_gt, trans_gt)
            view_matrix_pred = quaternion_and_translation_to_view_matrix(pred_rot, pred_trans)

            # --- Project Points --- 
            # Pass viewport params to projection
            kp2d_proj_from_gt = project_points(kp3d_world_gt, view_matrix_gt, proj_matrix, viewport_params)
            kp2d_proj_from_pred = project_points(kp3d_world_gt, view_matrix_pred, proj_matrix, viewport_params)

            # --- Create Plot --- 
            # Change layout to 2 rows, 3 columns
            fig = plt.figure(figsize=(24, 12)) # Adjusted figsize
            ax1 = fig.add_subplot(231, projection='3d') # 3D GT
            ax2 = fig.add_subplot(232) # 2D GT (Dataset)
            ax3 = fig.add_subplot(233) # 2D Projected (from GT transform) - Limited View
            # Use position 234 for the predicted limited view
            ax4 = fig.add_subplot(234) # 2D Projected (from Pred transform) - Limited View 
            ax5 = fig.add_subplot(235) # Interpolated 2D Proj (GT Transf)
            ax6 = fig.add_subplot(236) # Interpolated 2D Proj (Pred Transf)
            

            # --- Plotting --- 
            
            # Row 1
            # Plot 3D GT
            plot_skeleton_3d(ax1, kp3d_world_gt, joint_order, parent_map, "GT 3D Skeleton (World)")
            lims_3d = {'xlim': ax1.get_xlim(), 'ylim': ax1.get_ylim(), 'zlim': ax1.get_zlim()}

            # Plot 2D GT (Dataset)
            valid_kp2d_gt = kp2d_cam_gt[~np.isnan(kp2d_cam_gt).any(axis=1)]
            if valid_kp2d_gt.size > 0:
                min_xy = valid_kp2d_gt.min(axis=0); max_xy = valid_kp2d_gt.max(axis=0)
                center = (min_xy+max_xy)/2; range_ = (max_xy-min_xy).max()*0.6
                lims_2d_gt = {'xlim': (center[0]-range_, center[0]+range_), 'ylim':(center[1]-range_, center[1]+range_)}
            else:
                lims_2d_gt = None # Use auto-limits if no valid points
            plot_skeleton_2d(ax2, kp2d_cam_gt, joint_order, parent_map, "GT 2D Skeleton (Dataset Viewport)", color='red', limits=lims_2d_gt)

            # Plot 2D Projected from GT Transform (Limited View)
            plot_skeleton_2d(ax3, kp2d_proj_from_gt, joint_order, parent_map, "Projected 2D (GT Transf) - In Viewport", color='green', limits=lims_2d_gt) # Use same limits as 2D GT

            # Row 2
            # Plot 2D Projected from Predicted Transform (Limited View)
            plot_skeleton_2d(ax4, kp2d_proj_from_pred, joint_order, parent_map, "Projected 2D (Pred Transf) - In Viewport", color='purple', limits=lims_2d_gt) # Use same limits as 2D GT
            
            # Plot Interpolated 2D Projected from GT Transform
            plot_skeleton_2d_interpolated(ax5, kp2d_proj_from_gt, joint_order, parent_map, "Projected 2D Full (GT Transf)", color='green', limits_to_check=lims_2d_gt)

            # Plot Interpolated 2D Projected from Predicted Transform
            plot_skeleton_2d_interpolated(ax6, kp2d_proj_from_pred, joint_order, parent_map, "Projected 2D Full (Pred Transf)", color='purple', limits_to_check=lims_2d_gt)

            plt.suptitle(f"Epoch {epoch+1} - Sample {sample_idx} Visualization", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
            
            # Save the figure
            save_path = os.path.join(plot_base_dir, f"epoch_{epoch+1:03d}_sample_{sample_idx}.png")
            plt.savefig(save_path)
            plt.close(fig) # Close the figure to free memory
            # logger.info(f"Saved visualization to {save_path}") # Logged outside loop now

        except IndexError:
             logger.warning(f"Sample index {sample_idx} out of range for visualization. Max index: {len(dataloader)-1}")
        except Exception as e:
            logger.error(f"Error generating visualization for epoch {epoch+1}, sample {sample_idx}: {e}", exc_info=True)
    
    logger.info(f"Finished generating visualizations for epoch {epoch+1}.")

# --- Function to Print Sample Prediction ---

def print_sample_prediction(model, dataloader, sample_idx):
    """Prints the GT and predicted transform for a specific sample."""
    logger.info(f"--- Sample {sample_idx} Prediction --- ")
    try:
        sample = dataloader.__getitem__(sample_idx)
        quat_gt = sample['view_rot_quat']
        trans_gt = sample['view_trans']
        kp2d_quat_input = sample['kp2d_quat']
        joint_order = dataloader.joint_order

        # Get prediction
        model_input = tf.reshape(kp2d_quat_input, [1, len(joint_order), 1, 4])
        prediction = model(model_input, training=False)
        pred_rot = prediction[0, :4].numpy()
        pred_trans = prediction[0, 4:].numpy()

        logger.info(f"  Ground Truth Rotation (Quat wxyz): {quat_gt}")
        logger.info(f"  Predicted    Rotation (Quat wxyz): {pred_rot}")
        logger.info(f"  Ground Truth Translation (xyz):    {trans_gt}")
        logger.info(f"  Predicted    Translation (xyz):    {pred_trans}")

    except IndexError:
        logger.warning(f"Sample index {sample_idx} out of range for prediction printing.")
    except Exception as e:
        logger.error(f"Error getting prediction for sample {sample_idx}: {e}", exc_info=True)
    logger.info(f"------------------------------------")

def main(args):
    # Create output directories
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    
    # Load data
    print("Loading data...")
    projection_matrix = np.array([2.790552, 0., 0., 0.,
                                  0., 1.5696855, 0., 0.,
                                  0., 0., -1.0001999, -1.,
                                  0., 0., -0.20002, 0.], dtype=np.float32).reshape((4, 4))
    
    # Load routing config from main config file
    routing_config = config.get("routing", { "method": "dynamic", "iterations": 3 }) # Default if not found

    qcn_dataloader = QCNDataloader(args.train_data, args.val_data, projection_matrix=projection_matrix)
    train_dataset = qcn_dataloader.prepare_data(args.batch_size, is_training=True)
    val_dataset = qcn_dataloader.prepare_data(args.batch_size, is_training=False)
    
    # Determine dataset size for random sampling
    # Use validation set if large enough, otherwise training set
    vis_dataset_size = len(qcn_dataloader.val_data)
    if vis_dataset_size < NUM_PLOT_SAMPLES:
        logger.warning(f"Validation set size ({vis_dataset_size}) is smaller than num_plot_samples ({NUM_PLOT_SAMPLES}). Using training set for visualization sampling.")
        vis_dataset_size = len(qcn_dataloader.train_data)
        # Adjust dataloader access if using train data (though __getitem__ uses train_data by default)
        # Might need a flag in generate_epoch_visualization if strict separation is needed.

    # Initialize model, passing the routing config
    print("Initializing model...")
    model = QuaternionCapsNet(
        in_channels=4, 
        out_caps=1, 
        routing_config=routing_config # Pass routing config here
    )  
    
    # Build the model to infer shapes and print summary
    # Input shape for build: (B, NumJoints, QuatDim) -> (None, 31, 4) - after reshape
    # The initial x passed to call is (B, NumJoints, 1, QuatDim)
    build_input_shape = (args.batch_size, len(qcn_dataloader.joint_order), 4) # Shape used by Conv1D
    model.build(input_shape=(args.batch_size, len(qcn_dataloader.joint_order), 1, 4)) # Original input shape before reshape
    model.summary() # Print model summary
    
    # Setup optimizer with learning rate scheduling and gradient clipping
    initial_lr = args.learning_rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr, clipnorm=1.0) # Added clipnorm=1.0
    
    # Setup tensorboard logging
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = f'logs/{current_time}/train'
    val_log_dir = f'logs/{current_time}/val'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    val_summary_writer = tf.summary.create_file_writer(val_log_dir)
    
    # Training loops
    print("Starting training...")
    best_val_loss = float('inf')
    patience = args.patience
    patience_counter = 0
    current_lr = initial_lr # Track current LR for manual reduction
    
    for epoch in range(args.epochs):
        # Training
        train_loss = 0
        train_rot_loss = 0
        train_trans_loss = 0
        train_reproj_loss = 0
        num_batches = 0
        
        # Wrap train_dataset with tqdm for progress bar
        pbar = tqdm(train_dataset, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        # Get viewport once - assuming it's constant for all batches
        viewport_params_tf = tf.constant(VIEWPORT_PARAMS, dtype=tf.float32)
        for kp2d_quat, view_rot_quat, view_trans, kp2d_gt, kp3d_world, proj_matrix in pbar: # Updated loop with tqdm
            # Reshape input for model if needed
            batch_size = tf.shape(kp2d_quat)[0]
            # Reshape to match expected input shape for QuaternionCapsNet
            # Convert joint quaternions (B, 31, 4) to a 2D spatial representation (B, H, W, C)
            # where H*W = 31 (number of joints) and C = 4 (quaternion dimension)
            # ex:- reshape to (B, 31, 1, 4) which is effectively a 1D spatial array of quaternions
            x_input = tf.reshape(kp2d_quat, [batch_size, kp2d_quat.shape[1], 1, kp2d_quat.shape[2]])
            
            loss, rot_loss, trans_loss, reproj_loss = train_step(model, optimizer, x_input, view_rot_quat, view_trans, kp2d_gt, kp3d_world, proj_matrix, viewport_params_tf)
            train_loss += loss
            train_rot_loss += rot_loss
            train_trans_loss += trans_loss
            train_reproj_loss += reproj_loss # Accumulate reproj loss
            num_batches += 1
            
            # Update tqdm description with current losses
            pbar.set_postfix({
                'GeoLoss': f'{loss:.4f}',
                'RotLoss': f'{rot_loss:.4f}',
                'TransLoss': f'{trans_loss:.4f}',
                'ReprojLoss': f'{reproj_loss:.4f}'
            })
        
        train_loss /= num_batches
        train_rot_loss /= num_batches
        train_trans_loss /= num_batches
        train_reproj_loss /= num_batches
        
        # Validation
        val_loss = 0
        val_rot_loss = 0
        val_trans_loss = 0
        val_reproj_loss = 0
        num_val_batches = 0
        
        # Wrap val_dataset with tqdm
        pbar_val = tqdm(val_dataset, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
        # viewport_params_tf already defined
        for kp2d_quat, view_rot_quat, view_trans, kp2d_gt, kp3d_world, proj_matrix in pbar_val: # Updated loop with tqdm
            # Reshape input for model if needed
            batch_size = tf.shape(kp2d_quat)[0]
            # Convert joint quaternions (B, 31, 4) to a 2D spatial representation
            x_input = tf.reshape(kp2d_quat, [batch_size, kp2d_quat.shape[1], 1, kp2d_quat.shape[2]])
            
            loss, rot_loss, trans_loss, reproj_loss = val_step(model, x_input, view_rot_quat, view_trans, kp2d_gt, kp3d_world, proj_matrix, viewport_params_tf)
            val_loss += loss
            val_rot_loss += rot_loss
            val_trans_loss += trans_loss
            val_reproj_loss += reproj_loss # Accumulate reproj loss
            num_val_batches += 1
            
            # Update tqdm description for validation
            pbar_val.set_postfix({
                'GeoLoss': f'{loss:.4f}',
                'RotLoss': f'{rot_loss:.4f}',
                'TransLoss': f'{trans_loss:.4f}',
                'ReprojLoss': f'{reproj_loss:.4f}'
            })
        
        val_loss /= num_val_batches
        val_rot_loss /= num_val_batches
        val_trans_loss /= num_val_batches
        val_reproj_loss /= num_val_batches
        
        # Log metrics
        with train_summary_writer.as_default():
            tf.summary.scalar('geometric_loss', train_loss, step=epoch)
            tf.summary.scalar('rotation_loss', train_rot_loss, step=epoch)
            tf.summary.scalar('translation_loss', train_trans_loss, step=epoch)
            tf.summary.scalar('reprojection_loss', train_reproj_loss, step=epoch)
        
        with val_summary_writer.as_default():
            tf.summary.scalar('geometric_loss', val_loss, step=epoch)
            tf.summary.scalar('rotation_loss', val_rot_loss, step=epoch)
            tf.summary.scalar('translation_loss', val_trans_loss, step=epoch)
            tf.summary.scalar('reprojection_loss', val_reproj_loss, step=epoch)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Geo Loss: {train_loss:.4f}, Rot Loss: {train_rot_loss:.4f}, Trans Loss: {train_trans_loss:.4f}, Reproj Loss: {train_reproj_loss:.4f}")
        print(f"  Val Geo Loss: {val_loss:.4f}, Rot Loss: {val_rot_loss:.4f}, Trans Loss: {val_trans_loss:.4f}, Reproj Loss: {val_reproj_loss:.4f}")
        
        # --- Generate visualization periodically ---
        if PLOT_EVERY > 0 and vis_dataset_size > 0 and (epoch + 1) % PLOT_EVERY == 0:
            # Select random sample indices
            num_to_select = min(NUM_PLOT_SAMPLES, vis_dataset_size) # Ensure we don't ask for more samples than available
            vis_indices = random.sample(range(vis_dataset_size), num_to_select)
            # Pass the main config dictionary, checkpoint dir used as base save dir, and selected indices
            generate_epoch_visualization(model, qcn_dataloader, config, epoch, args.ckpt_dir, vis_indices)
        
        # Save model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0 # Reset patience
            model_path = os.path.join(args.ckpt_dir, f"model_epoch_{epoch+1:03d}.weights.h5") 
            model.save_weights(model_path)
            # Save best model separately with the correct format
            best_model_path = os.path.join(args.ckpt_dir, "best_model.weights.h5")
            model.save_weights(best_model_path)
            print(f"  Model saved to {model_path} (new best)")
        else:
            patience_counter += 1
            logger.info(f"  Val loss did not improve. Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                # Reduce LR if patience exceeded
                old_lr = optimizer.learning_rate.numpy()
                if old_lr > 1e-6: # Don't reduce LR indefinitely
                    new_lr = old_lr * 0.5 # Halve the learning rate
                    optimizer.learning_rate.assign(new_lr)
                    patience_counter = 0 # Reset patience after LR reduction
                    logger.warning(f"  Val loss plateaued for {patience} epochs. Reducing learning rate from {old_lr:.6f} to {new_lr:.6f}.")
                else:
                    logger.warning(f"  Learning rate already low ({old_lr:.6f}). Stopping training due to early stopping.")
                    break # Early stopping
    
    print("Training completed!")
    # Save final model with the correct format
    final_model_path = os.path.join(args.ckpt_dir, "final_model.weights.h5")
    model.save_weights(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # --- Print a random sample prediction before final eval ---
    if vis_dataset_size > 0: # Make sure there are samples
        pred_sample_idx = random.randint(0, vis_dataset_size - 1)
        print_sample_prediction(model, qcn_dataloader, pred_sample_idx)
    # ----------------------------------------------------------

    # Load best model for evaluation using the correct format
    model.load_weights(os.path.join(args.ckpt_dir, "best_model.weights.h5"))
    print(f"Loaded best model (val_loss: {best_val_loss:.4f}) for final evaluation")
    
    # Evaluate on test set
    if args.evaluate:
        print("Evaluating on validation set...")
        test_dataset = qcn_dataloader.prepare_data(args.batch_size, is_training=False)
        results = evaluate_model(model, test_dataset)
        
        print(f"Test Loss (Geometric): {results['test_loss']:.4f}")
        print(f"Component - Test Rotation Loss: {results['test_rot_loss']:.4f}")
        print(f"Component - Test Translation Loss: {results['test_trans_loss']:.4f}")
        print(f"Component - Test Reprojection Loss: {results['test_reproj_loss']:.4f}")
        
        # Visualize results
        visualize_dir = os.path.join(args.ckpt_dir, 'visualizations')
        visualize_results(results, visualize_dir)
        print(f"Visualizations saved to {visualize_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train QuaternionCapsNet for 3D pose estimation")
    parser.add_argument("--train_data", type=str, default="./data/train_data_20.json", help="Path to training data")
    parser.add_argument("--val_data", type=str, default="./data/val_data_20.json", help="Path to validation data")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation on validation set after training")
    
    args = parser.parse_args()
    main(args)
        