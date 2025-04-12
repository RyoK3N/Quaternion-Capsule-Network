import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import os

# Ensure the main project directory is in the path if running from a subfolder
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdataloader import QCNDataloader
from utils import get_parent_map # Assuming get_parent_map is sufficient

# --- Configuration ---
DATA_PATH = './data/train_data_20.json'
VAL_DATA_PATH = './data/val_data_20.json'
SAMPLE_INDEX = 1113 # Index of the sample to visualize

# Projection matrix (same as in qdataloader example)
PROJECTION_MATRIX = np.array([
    2.790552,  0.,         0.,         0.,
    0.,         1.5696855,  0.,         0.,
    0.,         0.,        -1.0001999, -1.,
    0.,         0.,        -0.20002,    0.
], dtype=np.float32).reshape((4, 4))

# Joint connectivity for plotting
PARENT_MAP = get_parent_map()

# --- Helper Functions ---

def quaternion_and_translation_to_view_matrix(quat_wxyz, trans_xyz):
    """Converts wxyz quaternion and xyz translation to a 4x4 view matrix."""
    # Ensure quaternion is normalized
    quat_wxyz = quat_wxyz / (np.linalg.norm(quat_wxyz) + 1e-8)
    # Scipy expects [x, y, z, w]
    quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    rotation_matrix = R.from_quat(quat_xyzw).as_matrix()

    view_matrix = np.eye(4)
    view_matrix[:3, :3] = rotation_matrix
    # Assuming standard convention where translation is in the last column
    # If your original vm had translation in the last row, adjust accordingly
    view_matrix[:3, 3] = trans_xyz
    return view_matrix

def project_points(points_3d, view_matrix, proj_matrix):
    """Projects 3D points (N, 3) to 2D (N, 2) using view and projection matrices."""
    num_points = points_3d.shape[0]
    # Convert to homogeneous coordinates (add 1)
    points_4d = np.hstack((points_3d, np.ones((num_points, 1))))

    # Transform: World -> View -> Clip space
    # Note: View matrix transforms world to view (camera) space.
    # Matrix multiplication order depends on convention (points as rows or columns).
    # Assuming points are row vectors: P_clip = P_world @ View.T @ Proj.T
    # Let's use the convention from the provided script: P_view = P_world_h @ View
    #                                                   P_clip = P_view @ Proj.T
    # This assumes View matrix includes translation in the last row.
    # Our reconstructed view matrix has translation in the last column.
    # Let's adjust the projection logic slightly for clarity assuming standard transforms.

    # World to View: P_view = R * P_world + t
    # -> Homogeneous: P_view_h = P_world_h @ ViewMatrix.T
    # (Where ViewMatrix has R in top-left 3x3, t in top-right 3x1, [0001] in last row)
    # Our view_matrix is constructed with t in the last column, so need transpose.

    points_view_h = points_4d @ view_matrix.T # Apply view transform

    # View to Clip: P_clip = P_view_h @ ProjMatrix.T
    points_clip_h = points_view_h @ proj_matrix.T # Apply projection

    # Perspective divide (ignore points with w near zero)
    # Add epsilon to avoid division by zero
    w = points_clip_h[:, 3:4]
    valid_idx = np.abs(w) > 1e-6
    points_ndc = np.full_like(points_clip_h[:, :2], np.nan) # Initialize with NaN
    points_ndc[valid_idx.flatten()] = points_clip_h[valid_idx.flatten(), :2] / w[valid_idx]

    # Assuming NDC is [-1, 1]. Convert to pixel space if needed, but loss uses this kind of space.
    # For visualization consistency with kp2d_gt, we might need viewport transform,
    # but let's plot the projected coordinates first.
    return points_ndc # Return coordinates in normalized device or clip space before viewport


def plot_skeleton_3d(ax, points_3d, joint_order, parent_map, title, color='blue', limits=None):
    """Plots a 3D skeleton."""
    ax.clear()
    joint_to_idx = {name: i for i, name in enumerate(joint_order)}

    # Plot points
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], color=color, s=30)

    # Plot bones
    for joint, parent in parent_map.items():
        if parent and joint in joint_to_idx and parent in joint_to_idx:
            idx_j = joint_to_idx[joint]
            idx_p = joint_to_idx[parent]
            ax.plot([points_3d[idx_p, 0], points_3d[idx_j, 0]],
                    [points_3d[idx_p, 1], points_3d[idx_j, 1]],
                    [points_3d[idx_p, 2], points_3d[idx_j, 2]],
                    color=color, linewidth=1.5)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    # Use limits if provided, otherwise calculate
    if limits:
         ax.set_xlim(limits['xlim'])
         ax.set_ylim(limits['ylim'])
         ax.set_zlim(limits['zlim'])
    else:
        max_range = np.array([points_3d[:, 0].max()-points_3d[:, 0].min(),
                              points_3d[:, 1].max()-points_3d[:, 1].min(),
                              points_3d[:, 2].max()-points_3d[:, 2].min()]).max() / 2.0
        mid_x = (points_3d[:, 0].max()+points_3d[:, 0].min()) * 0.5
        mid_y = (points_3d[:, 1].max()+points_3d[:, 1].min()) * 0.5
        mid_z = (points_3d[:, 2].max()+points_3d[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.view_init(elev=10., azim=-60) # Adjust view angle if needed
    ax.set_aspect('auto') # Use 'equal' if axes are comparable, 'auto' otherwise


def plot_skeleton_2d(ax, points_2d, joint_order, parent_map, title, color='red', limits=None):
    """Plots a 2D skeleton."""
    ax.clear()
    joint_to_idx = {name: i for i, name in enumerate(joint_order)}

    valid_pts = ~np.isnan(points_2d).any(axis=1)

    # Plot points
    ax.scatter(points_2d[valid_pts, 0], points_2d[valid_pts, 1], color=color, s=30)

    # Plot bones
    for joint, parent in parent_map.items():
        if parent and joint in joint_to_idx and parent in joint_to_idx:
            idx_j = joint_to_idx[joint]
            idx_p = joint_to_idx[parent]
            if valid_pts[idx_j] and valid_pts[idx_p]: # Only plot if both points are valid
                ax.plot([points_2d[idx_p, 0], points_2d[idx_j, 0]],
                        [points_2d[idx_p, 1], points_2d[idx_j, 1]],
                        color=color, linewidth=1.5)

    ax.set_xlabel("X (pixels/NDC)")
    ax.set_ylabel("Y (pixels/NDC)")
    ax.set_title(title)
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box') # Ensure correct aspect ratio for 2D
    if limits:
        ax.set_xlim(limits['xlim'])
        ax.set_ylim(limits['ylim'])


# --- Main Visualization ---
if __name__ == "__main__":
    print("Loading data...")
    # Use QCNDataloader to get the processed data including quaternions
    qcn_loader = QCNDataloader(DATA_PATH, VAL_DATA_PATH, projection_matrix=PROJECTION_MATRIX)
    sample = qcn_loader.__getitem__(SAMPLE_INDEX)

    kp3d_world = sample['kp3d_world']       # (N, 3) world coordinates
    kp2d_gt = sample['kp2d_gt']             # (N, 2) ground truth 2D camera coordinates (from dataset)
    quat_gt_wxyz = sample['view_rot_quat']  # (4,) GT rotation quaternion [w, x, y, z]
    trans_gt_xyz = sample['view_trans']     # (3,) GT translation vector

    joint_order = qcn_loader.joint_order    # List of joint names
    num_joints = len(joint_order)

    print(f"Visualizing sample index: {SAMPLE_INDEX}")
    print(f"Number of joints: {num_joints}")
    print(f"GT Quaternion (w,x,y,z): {quat_gt_wxyz}")
    print(f"GT Translation (x,y,z): {trans_gt_xyz}")

    # Reconstruct the ground truth view matrix from quaternion and translation
    # Note: Ensure the translation component placement matches the projection function's assumption
    view_matrix_gt = quaternion_and_translation_to_view_matrix(quat_gt_wxyz, trans_gt_xyz)
    print("\nReconstructed GT View Matrix:\n", view_matrix_gt)

    # Project the 3D GT points using the reconstructed GT view matrix
    kp2d_projected_from_gt = project_points(kp3d_world, view_matrix_gt, PROJECTION_MATRIX)
    print("\nProjected 2D points (using GT quat+trans):\n", kp2d_projected_from_gt[:5]) # Print first 5
    print("\nGround Truth 2D points (from dataset):\n", kp2d_gt[:5]) # Print first 5

    # --- Plotting ---
    fig = plt.figure(figsize=(18, 6))
    ax1 = fig.add_subplot(131, projection='3d')
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    # 1. Plot 3D Ground Truth Skeleton
    plot_skeleton_3d(ax1, kp3d_world, joint_order, PARENT_MAP, "3D Skeleton (World GT)")
    limits_3d = {'xlim': ax1.get_xlim(), 'ylim': ax1.get_ylim(), 'zlim': ax1.get_zlim()}


    # 2. Plot 2D Ground Truth Skeleton (from dataset)
    # Determine sensible limits based on data range
    valid_kp2d_gt = kp2d_gt[~np.isnan(kp2d_gt).any(axis=1)]
    if valid_kp2d_gt.size > 0:
         min_xy_gt = valid_kp2d_gt.min(axis=0)
         max_xy_gt = valid_kp2d_gt.max(axis=0)
         center_gt = (min_xy_gt + max_xy_gt) / 2
         range_gt = (max_xy_gt - min_xy_gt).max() * 0.6 # Add padding
         limits_2d_gt = {'xlim': (center_gt[0] - range_gt, center_gt[0] + range_gt),
                         'ylim': (center_gt[1] - range_gt, center_gt[1] + range_gt)}
         plot_skeleton_2d(ax2, kp2d_gt, joint_order, PARENT_MAP, "2D Skeleton (Camera GT - Dataset)", color='red', limits=limits_2d_gt)
    else:
         plot_skeleton_2d(ax2, kp2d_gt, joint_order, PARENT_MAP, "2D Skeleton (Camera GT - Dataset)", color='red')


    # 3. Plot 2D Projected Skeleton (from 3D GT + GT Transform)
    # Determine sensible limits
    valid_kp2d_proj = kp2d_projected_from_gt[~np.isnan(kp2d_projected_from_gt).any(axis=1)]
    if valid_kp2d_proj.size > 0:
        min_xy_proj = valid_kp2d_proj.min(axis=0)
        max_xy_proj = valid_kp2d_proj.max(axis=0)
        center_proj = (min_xy_proj + max_xy_proj) / 2
        range_proj = (max_xy_proj - min_xy_proj).max() * 0.6 # Add padding
        limits_2d_proj = {'xlim': (center_proj[0] - range_proj, center_proj[0] + range_proj),
                          'ylim': (center_proj[1] - range_proj, center_proj[1] + range_proj)}
        plot_skeleton_2d(ax3, kp2d_projected_from_gt, joint_order, PARENT_MAP, "2D Skeleton (Projected from 3D GT + GT Transf.)", color='green', limits=limits_2d_proj)
    else:
        plot_skeleton_2d(ax3, kp2d_projected_from_gt, joint_order, PARENT_MAP, "2D Skeleton (Projected from 3D GT + GT Transf.)", color='green')

    plt.tight_layout()
    plt.show()

    print("\nVisualization complete. Check the plots.")
    print("Compare Plot 2 (red) and Plot 3 (green). They should be similar if data/transforms are consistent.")
