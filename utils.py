import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R

def get_parent_map():
    """
    Return a dictionary mapping joint names to their parent joint indices.
    This is used for computing joint quaternions.
    """
    # Define the skeleton hierarchy
    parent_map = {
        'Head': 'Neck',
        'HeadEnd': 'Head',
        'Neck': 'Chest',
        'Chest': 'SpineMid',
        'SpineMid': 'SpineLow',
        'SpineLow': 'Hips',
        'Hips': None,  # Root joint
        'LeftShoulder': 'Chest',
        'LeftArm': 'LeftShoulder',
        'LeftForearm': 'LeftArm',
        'LeftHand': 'LeftForearm',
        'LeftFinger': 'LeftHand',
        'LeftFingerEnd': 'LeftFinger',
        'RightShoulder': 'Chest',
        'RightArm': 'RightShoulder',
        'RightForearm': 'RightArm',
        'RightHand': 'RightForearm',
        'RightFinger': 'RightHand',
        'RightFingerEnd': 'RightFinger',
        'LeftThigh': 'Hips',
        'LeftLeg': 'LeftThigh',
        'LeftFoot': 'LeftLeg',
        'LeftToe': 'LeftFoot',
        'LeftToeEnd': 'LeftToe',
        'LeftHeel': 'LeftFoot',
        'RightThigh': 'Hips',
        'RightLeg': 'RightThigh',
        'RightFoot': 'RightLeg',
        'RightToe': 'RightFoot',
        'RightToeEnd': 'RightToe',
        'RightHeel': 'RightFoot'
    }
    return parent_map

def view_matrix_to_quaternion(vm):
    """
    Convert a 4x4 view matrix to a quaternion representing rotation.
    Uses SVD to ensure the rotation matrix is orthogonal before conversion.
    
    Args:
        vm: A 4x4 view matrix in numpy array format.
        
    Returns:
        A quaternion as a numpy array [w, x, y, z].
    """
    # Extract the rotation part (upper-left 3x3 submatrix)
    rotation_matrix = vm[:3, :3]
    
    # Ensure the rotation matrix is orthogonal using SVD if needed
    try:
        # Try using the original matrix first
        r = R.from_matrix(rotation_matrix)
    except Exception:
        # If that fails, use SVD to find the closest orthogonal matrix
        U, _, Vt = np.linalg.svd(rotation_matrix, full_matrices=True)
        # The closest orthogonal matrix is U @ Vt
        orthogonal_rotation = U @ Vt
        
        # Ensure proper rotation (det = 1)
        if np.linalg.det(orthogonal_rotation) < 0:
            # If determinant is negative, we have a reflection, not a rotation
            # Flip the sign of the last column of U
            U[:, -1] = -U[:, -1]
            orthogonal_rotation = U @ Vt
            
        r = R.from_matrix(orthogonal_rotation)
    
    # Get quaternion [x, y, z, w]
    quat_xyzw = r.as_quat()
    
    # Convert to [w, x, y, z] format
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

def compute_joint_quaternions_3d(keypoints_3d, joint_order, parent_map, ref=np.array([0, 1, 0])):
    """
    Compute quaternions for each joint in 3D space.
    
    Args:
        keypoints_3d: 3D keypoints as a (N, 3) array
        joint_order: List of joint names corresponding to the keypoints
        parent_map: Dictionary mapping joint names to parent joint names
        ref: Reference direction vector
        
    Returns:
        Array of quaternions for each joint
    """
    N = len(joint_order)
    quaternions = np.zeros((N, 4))
    
    # Create index mapping from joint name to index
    joint_to_idx = {joint: i for i, joint in enumerate(joint_order)}
    
    for i, joint in enumerate(joint_order):
        parent_name = parent_map.get(joint)
        if parent_name is None:
            # Root joint, use identity quaternion
            quaternions[i] = np.array([1, 0, 0, 0])  # w, x, y, z
            continue
            
        parent_idx = joint_to_idx[parent_name]
        
        # Compute bone direction
        joint_pos = keypoints_3d[i]
        parent_pos = keypoints_3d[parent_idx]
        
        # Skip if joint and parent are at the same position
        if np.allclose(joint_pos, parent_pos):
            quaternions[i] = np.array([1, 0, 0, 0])
            continue
            
        # Compute bone direction vector
        bone_vec = joint_pos - parent_pos
        bone_vec = bone_vec / (np.linalg.norm(bone_vec) + 1e-8)
        
        # Compute rotation from reference to bone direction
        # Find axis and angle between ref and bone_vec
        axis = np.cross(ref, bone_vec)
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm < 1e-8:
            # Vectors are parallel or anti-parallel
            if np.dot(ref, bone_vec) > 0:
                # Same direction
                quaternions[i] = np.array([1, 0, 0, 0])  # Identity quaternion
            else:
                # Opposite direction - 180 degree rotation
                # Choose a perpendicular axis
                if not np.allclose(ref, [1, 0, 0]) and not np.allclose(ref, [-1, 0, 0]):
                    perp_axis = np.cross(ref, [1, 0, 0])
                else:
                    perp_axis = np.cross(ref, [0, 1, 0])
                perp_axis = perp_axis / (np.linalg.norm(perp_axis) + 1e-8)
                
                # Create rotation of 180 degrees around perp_axis
                r = R.from_rotvec(np.pi * perp_axis)
                quat_xyzw = r.as_quat()
                quaternions[i] = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        else:
            # Normal case
            axis = axis / axis_norm
            angle = np.arccos(np.clip(np.dot(ref, bone_vec), -1.0, 1.0))
            
            # Create rotation
            r = R.from_rotvec(angle * axis)
            quat_xyzw = r.as_quat()
            quaternions[i] = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            
    return quaternions

def compute_joint_quaternions_2d(keypoints_2d, joint_order, parent_map, ref=np.array([1, 0])):
    """
    Compute quaternions for each joint in 2D space, representing rotation around Z-axis.
    
    Args:
        keypoints_2d: 2D keypoints as a (N, 2) array
        joint_order: List of joint names corresponding to the keypoints
        parent_map: Dictionary mapping joint names to parent joint names
        ref: Reference direction vector in 2D
        
    Returns:
        Array of quaternions for each joint (rotation around Z-axis)
    """
    N = len(joint_order)
    quaternions = np.zeros((N, 4))
    
    # Create index mapping from joint name to index
    joint_to_idx = {joint: i for i, joint in enumerate(joint_order)}
    
    for i, joint in enumerate(joint_order):
        parent_name = parent_map.get(joint)
        if parent_name is None:
            # Root joint, use identity quaternion
            quaternions[i] = np.array([1, 0, 0, 0])  # w, x, y, z
            continue
            
        parent_idx = joint_to_idx[parent_name]
        
        # Compute bone direction
        joint_pos = keypoints_2d[i]
        parent_pos = keypoints_2d[parent_idx]
        
        # Skip if joint and parent are at the same position
        if np.allclose(joint_pos, parent_pos):
            quaternions[i] = np.array([1, 0, 0, 0])
            continue
            
        # Compute bone direction vector
        bone_vec = joint_pos - parent_pos
        bone_vec = bone_vec / (np.linalg.norm(bone_vec) + 1e-8)
        
        # Compute angle between reference vector and bone vector
        cos_angle = np.dot(ref, bone_vec)
        sin_angle = np.cross(ref, bone_vec)  # Z-component of cross product
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        
        # If sin_angle is negative, the angle is negative
        if sin_angle < 0:
            angle = -angle
            
        # Create quaternion for rotation around Z-axis using scipy
        r = R.from_rotvec([0, 0, angle])
        quat_xyzw = r.as_quat()
        
        # Convert from scipy's [x, y, z, w] to our [w, x, y, z] format
        quaternions[i] = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
            
    return quaternions

def plot_figs_to_png(history, title, save_path):
    """
    Plot training and validation metrics and save to PNG files.
    
    Args:
        history: Training history dictionary with keys 'loss', 'val_loss', etc.
        title: Plot title
        save_path: Path to save the output PNG file
    """
    plt.figure(figsize=(12, 4))
    
    # Create subplots
    plt.subplot(1, 3, 1)
    plt.plot(history['loss'], label='Train')
    plt.plot(history['val_loss'], label='Val')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history['rotation_loss'], label='Train')
    plt.plot(history['val_rotation_loss'], label='Val')
    plt.title(f'{title} - Rotation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history['translation_loss'], label='Train')
    plt.plot(history['val_translation_loss'], label='Val')
    plt.title(f'{title} - Translation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close() 