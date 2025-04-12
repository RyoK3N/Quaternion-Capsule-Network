import tensorflow as tf
import numpy as np
from utils import view_matrix_to_quaternion

def qrotation_loss(y_true, y_pred):
    """Calculate geodesic distance between two quaternions, accounting for q and -q being the same rotation.
    
    Args:
        y_true: Ground truth quaternions in [w, x, y, z] format
        y_pred: Predicted quaternions in [w, x, y, z] format
        
    Returns:
        Mean geodesic distance between quaternions
    """
    # Normalize quaternions to ensure they are unit quaternions
    q1_norm = tf.nn.l2_normalize(y_true, axis=-1)
    q2_norm = tf.nn.l2_normalize(y_pred, axis=-1)
    
    # Calculate dot product between quaternions
    dot_product = tf.reduce_sum(q1_norm * q2_norm, axis=-1)
    
    # Clamp dot product to valid range [-1, 1]
    dot_product = tf.clip_by_value(tf.abs(dot_product), 0.0, 1.0)
    
    # Calculate geodesic distance (angle in radians)
    angle = 2.0 * tf.acos(dot_product)
    
    return tf.reduce_mean(angle)

def qtranslation_loss(y_true, y_pred):
    """Calculate L1 loss between translation vectors.
    
    Args:
        y_true: Ground truth translation vector
        y_pred: Predicted translation vector
        
    Returns:
        Mean absolute error between translations
    """
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def qreprojection_loss(kp2d_gt, kp3d_world, quat_gt, trans_gt, quat_pred, trans_pred, proj_matrix, viewport):
    """Calculate reprojection loss using quaternion rotations, proj matrix, and viewport.
    
    Args:
        kp2d_gt: Ground truth 2D keypoints (Screen Coords), shape (batch_size, num_points, 2).
        kp3d_world: Ground truth 3D keypoints in world coordinates
        quat_gt: Ground truth quaternion
        trans_gt: Ground truth translation
        quat_pred: Predicted quaternion
        trans_pred: Predicted translation
        proj_matrix: Projection matrix (4x4)
        viewport: Tuple/List containing [x, y, width, height] for viewport transform.
        
    Returns:
        Mean reprojection error (scalar tensor).
    """
    batch_size = tf.shape(kp2d_gt)[0]
    num_joints = tf.shape(kp3d_world)[1]
    
    # Ensure quaternion inputs are float32
    quat_gt = tf.cast(quat_gt, tf.float32)
    trans_gt = tf.cast(trans_gt, tf.float32)
    quat_pred = tf.cast(quat_pred, tf.float32)
    trans_pred = tf.cast(trans_pred, tf.float32)
    kp3d_world = tf.cast(kp3d_world, tf.float32)
    kp2d_gt = tf.cast(kp2d_gt, tf.float32)
    proj_matrix = tf.cast(proj_matrix, tf.float32) # Cast proj_matrix too

    # Normalize quaternions
    quat_gt_norm = tf.nn.l2_normalize(quat_gt, axis=-1)
    quat_pred_norm = tf.nn.l2_normalize(quat_pred, axis=-1)
    
    # Convert quaternions to rotation matrices
    # For simplicity, we'll use the quaternion to create rotation matrices directly
    w_gt, x_gt, y_gt, z_gt = tf.unstack(quat_gt_norm, axis=-1)
    w_pred, x_pred, y_pred, z_pred = tf.unstack(quat_pred_norm, axis=-1)
    
    # Quaternion to rotation matrix (ground truth)
    xx_gt = x_gt * x_gt
    yy_gt = y_gt * y_gt
    zz_gt = z_gt * z_gt
    xy_gt = x_gt * y_gt
    xz_gt = x_gt * z_gt
    yz_gt = y_gt * z_gt
    wx_gt = w_gt * x_gt
    wy_gt = w_gt * y_gt
    wz_gt = w_gt * z_gt
    
    r00_gt = 1 - 2 * (yy_gt + zz_gt)
    r01_gt = 2 * (xy_gt - wz_gt)
    r02_gt = 2 * (xz_gt + wy_gt)
    r10_gt = 2 * (xy_gt + wz_gt)
    r11_gt = 1 - 2 * (xx_gt + zz_gt)
    r12_gt = 2 * (yz_gt - wx_gt)
    r20_gt = 2 * (xz_gt - wy_gt)
    r21_gt = 2 * (yz_gt + wx_gt)
    r22_gt = 1 - 2 * (xx_gt + yy_gt)
    
    # Create rotation matrix (ground truth)
    rot_gt = tf.stack([
        tf.stack([r00_gt, r01_gt, r02_gt], axis=-1),
        tf.stack([r10_gt, r11_gt, r12_gt], axis=-1),
        tf.stack([r20_gt, r21_gt, r22_gt], axis=-1)
    ], axis=-2)
    
    # Quaternion to rotation matrix (predicted)
    xx_pred = x_pred * x_pred
    yy_pred = y_pred * y_pred
    zz_pred = z_pred * z_pred
    xy_pred = x_pred * y_pred
    xz_pred = x_pred * z_pred
    yz_pred = y_pred * z_pred
    wx_pred = w_pred * x_pred
    wy_pred = w_pred * y_pred
    wz_pred = w_pred * z_pred
    
    r00_pred = 1 - 2 * (yy_pred + zz_pred)
    r01_pred = 2 * (xy_pred - wz_pred)
    r02_pred = 2 * (xz_pred + wy_pred)
    r10_pred = 2 * (xy_pred + wz_pred)
    r11_pred = 1 - 2 * (xx_pred + zz_pred)
    r12_pred = 2 * (yz_pred - wx_pred)
    r20_pred = 2 * (xz_pred - wy_pred)
    r21_pred = 2 * (yz_pred + wx_pred)
    r22_pred = 1 - 2 * (xx_pred + yy_pred)
    
    # Create rotation matrix (predicted)
    rot_pred = tf.stack([
        tf.stack([r00_pred, r01_pred, r02_pred], axis=-1),
        tf.stack([r10_pred, r11_pred, r12_pred], axis=-1),
        tf.stack([r20_pred, r21_pred, r22_pred], axis=-1)
    ], axis=-2)
    
    # --- Project 3D points using full matrix transformations (TensorFlow) ---
    
    # Add homogeneous coordinate to 3D world points
    ones = tf.ones([batch_size, num_joints, 1], dtype=tf.float32)
    kp3d_world_h = tf.concat([kp3d_world, ones], axis=-1) # Shape: [B, N, 4]

    # --- Ground Truth Projection ---
    # Construct 4x4 view matrix for GT
    view_matrix_gt = tf.eye(4, batch_shape=[batch_size], dtype=tf.float32)
    # Need to assign rotation and translation carefully to match matrix structure
    # Assuming standard view matrix: [ R | t ]
    #                                [ 0 | 1 ]
    padding_gt = tf.zeros([batch_size, 1, 3], dtype=tf.float32) 
    bottom_row_gt = tf.constant([[0.0, 0.0, 0.0, 1.0]], dtype=tf.float32)
    bottom_row_gt = tf.tile(bottom_row_gt, [batch_size, 1]) # Shape [B, 4]
    
    # Assemble R and t into top-left 3x4
    rot_gt_batched = tf.cast(rot_gt, tf.float32)
    trans_gt_batched = tf.expand_dims(tf.cast(trans_gt, tf.float32), axis=-1) # Shape [B, 3, 1]
    top_part_gt = tf.concat([rot_gt_batched, trans_gt_batched], axis=2) # Shape [B, 3, 4]
    # Assemble View Matrix with translation in the last row
    # Top 3 rows: [R | 0] -> needs modification
    zeros_t = tf.zeros_like(trans_gt_batched)
    top_3x4_gt = tf.concat([rot_gt_batched, zeros_t], axis=2) # [B, 3, 4]
    # Last row: [t | 1]
    trans_gt_row = tf.transpose(trans_gt_batched, perm=[0, 2, 1]) # [B, 1, 3]
    ones_row = tf.ones([batch_size, 1, 1], dtype=tf.float32)
    bottom_row_gt = tf.concat([trans_gt_row, ones_row], axis=2) # [B, 1, 4]
    view_matrix_gt = tf.concat([top_3x4_gt, bottom_row_gt], axis=1) # [B, 4, 4]

    # Project GT: P_clip_h = P_world_h @ View @ Proj
    # No transpose needed now if matrices are OpenGL style
    # view_matrix_gt_t = tf.transpose(view_matrix_gt, perm=[0, 2, 1])
    # proj_matrix_t = tf.transpose(proj_matrix, perm=[0, 2, 1])
    
    # Ensure proj_matrix is batched
    if len(tf.shape(proj_matrix)) == 2:
        proj_matrix_batched = tf.tile(tf.expand_dims(proj_matrix, 0), [batch_size, 1, 1])
    else:
        proj_matrix_batched = proj_matrix
    # proj_matrix_t = tf.transpose(proj_matrix_batched, perm=[0, 2, 1])
        
    # Apply transforms (No Transpose)
    kp_view_h_gt = tf.matmul(kp3d_world_h, view_matrix_gt)
    kp_clip_h_gt = tf.matmul(kp_view_h_gt, proj_matrix_batched)

    # Perspective divide for GT
    w_gt = kp_clip_h_gt[..., 3:4]
    # Avoid division by zero/small numbers
    w_gt_safe = w_gt + tf.keras.backend.epsilon() * tf.sign(w_gt) + tf.keras.backend.epsilon()
    kp_ndc_gt = kp_clip_h_gt[..., :2] / w_gt_safe

    # --- Apply Viewport Transform GT ---
    viewport_x, viewport_y, viewport_width, viewport_height = tf.unstack(tf.cast(viewport, tf.float32))
    x_ndc_gt = kp_ndc_gt[..., 0]
    y_ndc_gt = kp_ndc_gt[..., 1]
    x_screen_gt = (x_ndc_gt * 0.5 + 0.5) * viewport_width + viewport_x
    y_screen_gt = (y_ndc_gt * 0.5 + 0.5) * viewport_height + viewport_y # Assuming Y points down in screen
    kp2d_proj_gt_screen = tf.stack([x_screen_gt, y_screen_gt], axis=-1)

    # --- Predicted Projection (Repeat process) ---
    # Construct 4x4 view matrix for prediction (translation in last row)
    padding_pred = tf.zeros([batch_size, 1, 3], dtype=tf.float32)
    bottom_row_pred = tf.constant([[0.0, 0.0, 0.0, 1.0]], dtype=tf.float32)
    bottom_row_pred = tf.tile(bottom_row_pred, [batch_size, 1])
    rot_pred_batched = tf.cast(rot_pred, tf.float32)
    trans_pred_batched = tf.expand_dims(tf.cast(trans_pred, tf.float32), axis=-1)
    zeros_t_pred = tf.zeros_like(trans_pred_batched)
    top_3x4_pred = tf.concat([rot_pred_batched, zeros_t_pred], axis=2)
    trans_pred_row = tf.transpose(trans_pred_batched, perm=[0, 2, 1])
    ones_row_pred = tf.ones([batch_size, 1, 1], dtype=tf.float32)
    bottom_row_pred = tf.concat([trans_pred_row, ones_row_pred], axis=2)
    view_matrix_pred = tf.concat([top_3x4_pred, bottom_row_pred], axis=1)

    # Project Predicted: P_clip_h = P_world_h @ View @ Proj
    # view_matrix_pred_t = tf.transpose(view_matrix_pred, perm=[0, 2, 1])
    # proj_matrix_batched is same as before
    
    kp_view_h_pred = tf.matmul(kp3d_world_h, view_matrix_pred)
    kp_clip_h_pred = tf.matmul(kp_view_h_pred, proj_matrix_batched)

    # Perspective divide for Prediction
    w_pred = kp_clip_h_pred[..., 3:4]
    w_pred_safe = w_pred + tf.keras.backend.epsilon() * tf.sign(w_pred) + tf.keras.backend.epsilon()
    kp_ndc_pred = kp_clip_h_pred[..., :2] / w_pred_safe

    # --- Apply Viewport Transform Prediction ---
    # Viewport params already unpacked
    x_ndc_pred = kp_ndc_pred[..., 0]
    y_ndc_pred = kp_ndc_pred[..., 1]
    x_screen_pred = (x_ndc_pred * 0.5 + 0.5) * viewport_width + viewport_x
    y_screen_pred = (y_ndc_pred * 0.5 + 0.5) * viewport_height + viewport_y # Assuming Y points down
    kp2d_proj_pred_screen = tf.stack([x_screen_pred, y_screen_pred], axis=-1)

    # --- Normalize Coordinates before Loss Calculation ---
    # Normalize based on viewport dimensions to get values roughly in [0, 1]
    # Avoid division by zero if width/height are somehow zero
    safe_width = tf.maximum(viewport_width, 1e-6)
    safe_height = tf.maximum(viewport_height, 1e-6)
    
    # Normalize predicted points
    kp2d_pred_norm_x = (kp2d_proj_pred_screen[..., 0] - viewport_x) / safe_width
    kp2d_pred_norm_y = (kp2d_proj_pred_screen[..., 1] - viewport_y) / safe_height
    kp2d_pred_norm = tf.stack([kp2d_pred_norm_x, kp2d_pred_norm_y], axis=-1)

    # Normalize ground truth points
    kp2d_gt = tf.cast(kp2d_gt, tf.float32)
    kp2d_gt_norm_x = (kp2d_gt[..., 0] - viewport_x) / safe_width
    kp2d_gt_norm_y = (kp2d_gt[..., 1] - viewport_y) / safe_height
    kp2d_gt_norm = tf.stack([kp2d_gt_norm_x, kp2d_gt_norm_y], axis=-1)

    # --- Calculate Reprojection Error on NORMALIZED points --- 
    # Use L2 distance between normalized points
    reprojection_error = tf.sqrt(tf.reduce_sum(tf.square(kp2d_pred_norm - kp2d_gt_norm), axis=-1))
    
    # Return the mean error across all joints and batch items
    mean_reproj_error = tf.reduce_mean(reprojection_error)
    
    return mean_reproj_error

def qgeometric_loss(kp2d_gt, kp3d_world, quat_true, trans_true, quat_pred, trans_pred, proj_matrix,
                   viewport,
                   alpha=1.0, beta=0.1, gamma=0.1, unit_weight=0.1):
    """Combined geometric loss using quaternion rotations.
    
    Args:
        kp2d_gt: Ground truth 2D keypoints
        kp3d_world: Ground truth 3D keypoints in world coordinates
        quat_true: Ground truth quaternions
        trans_true: Ground truth translations
        quat_pred: Predicted quaternions
        trans_pred: Predicted translations
        proj_matrix: Projection matrix (4x4)
        viewport: Tuple/List containing [x, y, width, height].
        alpha: Weight for rotation loss
        beta: Weight for translation loss
        gamma: Weight for reprojection loss
        unit_weight: Weight for unit quaternion constraint
        
    Returns:
        Combined geometric loss
    """
    # Calculate component losses
    rot_loss = qrotation_loss(quat_true, quat_pred)
    trans_loss = qtranslation_loss(trans_true, trans_pred)
    reproj_loss = qreprojection_loss(kp2d_gt, kp3d_world, quat_true, trans_true, quat_pred, trans_pred, proj_matrix, viewport)
    
    # Normalize quaternions constraint (quaternions should be unit length)
    norm_q_pred = tf.norm(quat_pred, axis=-1)
    unit_constraint = tf.reduce_mean(tf.abs(norm_q_pred - 1.0))
    
    # Combined loss
    total_loss = alpha * rot_loss + beta * trans_loss + gamma * reproj_loss + unit_weight * unit_constraint
    
    return total_loss