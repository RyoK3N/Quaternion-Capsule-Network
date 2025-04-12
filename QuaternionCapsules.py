"""
QuaternionCapsules.py
=====================
A TensorFlow implementation of Quaternion Capsule Networks (QCN) This version 
accept 2D joint quaternions as input and produce camera transformations
(rotation quaternion + translation). It uses branched residual blocks for pose
and activation feature extraction, then forms primary quaternion capsules, and
applies quaternion routing for higher-level capsules.

"""

import tensorflow as tf
import numpy as np
from ResidualBlocks import BasicPreActResBlock
from RoutingMethods import get_routing_method # Import the factory function
# from modules import QuaternionLayer
# from Routing_Methods import EMRouting

class BranchedResExtractor(tf.keras.layers.Layer):
    """
    Extracts separate features for 'pose' and 'activation' paths as done in
    the official QCN. Each path uses residual blocks, then merges.
    """
    def __init__(self, in_channels_pose=2, in_channels_act=2, mid_channels=32, out_channels=64, stride=1):
        super().__init__()
        # Pose path
        self.pose_block1 = BasicPreActResBlock(in_channels=in_channels_pose, out_channels=mid_channels, stride=1)
        self.pose_block2 = BasicPreActResBlock(in_channels=mid_channels, out_channels=out_channels, stride=1)
        # Activation path
        self.act_block1 = BasicPreActResBlock(in_channels=in_channels_act, out_channels=mid_channels, stride=1)

    def call(self, x_pose, x_act, training=False):
        pose_features = self.pose_block2(self.pose_block1(x_pose, training=training), training=training)
        act_features = self.act_block1(x_act, training=training)
        return pose_features, act_features


class PrimaryQuatCaps(tf.keras.layers.Layer):
    """
    Primary Quaternion Capsules layer: branches to produce pose quaternions
    and activation from separate feature maps, then merges them.
    """
    def __init__(self, out_caps=32, quat_dims=3):
        super().__init__()
        self.quat_dims = quat_dims
        self.out_caps = out_caps
        self.pose_conv = tf.keras.layers.Conv2D(
            filters=out_caps * quat_dims,
            kernel_size=1,
            strides=1,
            padding='valid'
        )
        self.act_conv = tf.keras.layers.Conv2D(
            filters=out_caps,
            kernel_size=1,
            strides=1,
            padding='valid'
        )
        self.pose_bn = tf.keras.layers.BatchNormalization()
        self.act_bn = tf.keras.layers.BatchNormalization()

    def call(self, pose_feats, act_feats, training=False):
        M = self.pose_bn(self.pose_conv(pose_feats), training=training)
        A = tf.nn.sigmoid(self.act_bn(self.act_conv(act_feats), training=training))
        # Reshape to [B, H, W, out_caps, quat_dims]
        # and [B, H, W, out_caps] for activation
        shape_pose = tf.shape(M)
        shape_act = tf.shape(A)
        # Suppose pose is (B, H, W, out_caps * quat_dims)
        M = tf.reshape(M, [shape_pose[0], shape_pose[1], shape_pose[2], self.out_caps, self.quat_dims])
        # Similarly A is (B, H, W, out_caps)
        return M, A


class ConvQuaternionCapsLayer(tf.keras.layers.Layer):
    """
    Example of a convolutional quaternion capsule layer that handles
    quaternion-based EM-routing, as per the official QCN. 
    (Placeholder for actual quaternion routing logic.)
    """
    def __init__(self, kernel_size=(1,1), stride=(1,1), in_caps=32, out_caps=32, quat_dims=3, routing_iters=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.routing_iters = routing_iters
        # self.routing = EMRouting(self.routing_iters)  # If using official QCN routing
        # Example trainable quaternion transform
        self.conv_pose = tf.keras.layers.Conv2D(
            filters=out_caps * quat_dims,
            kernel_size=kernel_size,
            strides=stride,
            padding='same',
            activation=None
        )
        self.conv_act = tf.keras.layers.Conv2D(
            filters=out_caps,
            kernel_size=kernel_size,
            strides=stride,
            padding='same',
            activation=None
        )

    def call(self, pose_in, act_in, training=False):
        # Suppose pose_in shape: [B, H, W, in_caps, quat_dims]
        # Flatten in_caps to channels for simplicity => (B, H, W, in_caps*quat_dims)
        b, h, w, ic, qd = tf.unstack(tf.shape(pose_in), num=5)
        pose_reshaped = tf.reshape(pose_in, [b, h, w, ic*qd])
        # Simple conv
        pose_out = self.conv_pose(pose_reshaped, training=training)
        act_out = self.conv_act(tf.reshape(act_in, [b, h, w, ic]), training=training)

        # Reshape back to capsules
        shape_pose_out = tf.shape(pose_out)
        oh, ow = shape_pose_out[1], shape_pose_out[2]
        # (B, oh, ow, out_caps*quat_dims)
        # => (B, oh, ow, out_caps, quat_dims)
        pose_out = tf.reshape(pose_out, [b, oh, ow, -1, qd])
        act_out = tf.reshape(act_out, [b, oh, ow, -1])

        # Apply transformation to input capsules (votes)
        # Pose votes u_hat = W_ij * u_i 
        # Input shapes: pose_in [B, H, W, in_caps, quat_dims], act_in [B, H, W, in_caps]
        b, h, w, ic, qd = tf.unstack(tf.shape(pose_in), num=5)
        
        # Reshape inputs for Conv2D
        # Pose: Treat InCaps * QuatDims as channels -> [B, H, W, IC*QD]
        pose_reshaped_for_conv = tf.reshape(pose_in, [b, h, w, ic * qd])
        # Act: Treat InCaps as channels -> [B, H, W, IC]
        act_reshaped_for_conv = tf.reshape(act_in, [b, h, w, ic])

        # Apply convolutions to generate votes for pose and activation predictions
        pose_votes_flat = self.conv_pose(pose_reshaped_for_conv) # [B, OH, OW, OC*QD]
        act_pred_flat = self.conv_act(act_reshaped_for_conv) # [B, OH, OW, OC]

        # Reshape votes back to capsule format for routing
        # Pose votes: [B, OH, OW, OutCaps, QuatDims]
        # Activation prediction (used differently depending on routing)
        oh, ow = tf.shape(pose_votes_flat)[1], tf.shape(pose_votes_flat)[2]
        pose_votes = tf.reshape(pose_votes_flat, [b, oh, ow, self.out_caps, self.quat_dims])
        act_pred = tf.reshape(act_pred_flat, [b, oh, ow, self.out_caps])

        # --- Apply Routing --- 
        # Note: Routing expects votes and input activations.
        # For conv layers, routing is typically applied spatially.
        # For simplicity here, we'll route globally after conv, 
        # or assume routing logic handles spatial dims if adapted.
        # This example assumes routing acts on the final dimensions.
        
        # Reshape activation input for routing: [B, H*W*in_caps]
        # Reshape pose votes for routing: [B, H*W*in_caps, out_caps, quat_dims]
        # This part needs careful adaptation depending on how routing handles spatial ConvCaps
        # Placeholder: Directly return conv output for now, routing needs more specific impl.
        
        # --- Apply Routing (Simpler: Assume routing takes feature maps) ---
        # We need input activations reshaped to match votes spatial structure if needed
        # e.g., Tile activations_in if routing is per spatial location
        
        # For demonstration, let's assume routing works on the final output maps directly
        # This is NOT standard capsule routing but shows integration.
        # Proper routing requires handling votes from each lower cap to each higher cap. 
        
        # If using a routing layer instance passed during init:
        if self.routing:
             # Prepare inputs for routing (Needs adaptation based on routing layer requirements)
             # This example passes the direct conv outputs, which isn't standard routing input
             # Proper implementation would involve transforming pose_in capsules to votes first.
             poses_out, act_out = self.routing(pose_votes, act_pred) # Simplified call
        else: # Fallback if no routing layer provided
             poses_out = tf.nn.l2_normalize(pose_votes, axis=-1)
             act_out = tf.nn.sigmoid(act_pred)

        return poses_out, act_out


class FCQuaternionCaps(tf.keras.layers.Layer):
    """
    Fully-connected quaternion capsule layer that aggregates from
    all spatial capsules, then routes to final output capsules
    (e.g., num_classes or final transform parameters).
    Accepts a routing layer instance.
    """
    def __init__(self, in_caps, out_caps, quat_dims=4, routing_layer=None):
        super().__init__()
        self.in_caps = in_caps # Number of input capsule types
        self.out_caps = out_caps # Number of output capsules
        self.quat_dims = quat_dims # Dimension of the pose vector (4 for quaternion)
        self.routing_layer = routing_layer # Instance of a routing method

        # Transformation matrix W_ij: applies unique transform from each input cap i to each output cap j
        # Shape: [InCaps, OutCaps, QuatDims, QuatDims]
        # We use Dense for simplicity: each output cap j gets votes from all input caps i
        # Input to Dense: [B, TotalInCaps, QuatDims]
        # Output from Dense: [B, TotalInCaps, OutCaps * QuatDims]
        self.fc_pose_votes = tf.keras.layers.Dense(out_caps * quat_dims)
        # Activation prediction is usually simpler, maybe direct Dense
        self.fc_act_pred = tf.keras.layers.Dense(out_caps)

    def call(self, pose_in, act_in, training=False):
        # Flatten spatial dims and combine with input capsule types
        # pose_in shape: [B, H, W, InCaps, QuatDims]
        # act_in shape: [B, H, W, InCaps]
        pose_shape = tf.shape(pose_in)
        b = pose_shape[0]
        h = pose_shape[1]
        w = pose_shape[2]
        ic = tf.cast(self.in_caps, tf.int32) # Use self.in_caps
        qd = tf.cast(self.quat_dims, tf.int32)

        # Flatten H, W, InCaps dims together -> [B, H*W*InCaps, QuatDims]
        # Corrected: Flatten NumJoints (h) and InCaps (w) dims -> [B, NumJoints * InCaps, QuatDims]
        total_in_caps_flattened = h * w
        pose_flat = tf.reshape(pose_in, [b, total_in_caps_flattened, qd])
        # Flatten activations -> [B, H*W*InCaps]
        # Corrected: Flatten NumJoints (h) and InCaps (w) dims -> [B, NumJoints * InCaps]
        act_flat = tf.reshape(act_in, [b, total_in_caps_flattened])

        # --- Generate Votes ---
        # Apply Dense layer to generate votes for each output capsule
        # Input: [B, TotalInCaps, QuatDims]
        # Output: [B, TotalInCaps, OutCaps * QuatDims]
        pose_votes_flat = self.fc_pose_votes(pose_flat)

        # Reshape votes: [B, TotalInCaps, OutCaps, QuatDims]
        # total_in_caps = h * w * ic # Original incorrect calculation
        # Corrected: Use the already calculated flattened dimension size
        pose_votes = tf.reshape(pose_votes_flat, [b, total_in_caps_flattened, self.out_caps, qd])

        # Activation prediction (can be simpler)
        # Input: [B, TotalInCaps] -> Output: [B, TotalInCaps, OutCaps]
        # act_pred_flat = self.fc_act_pred(act_flat) 
        # act_pred = tf.reshape(act_pred_flat, [b, total_in_caps, self.out_caps])
        # For routing, we need the flattened lower-level activations

        # --- Apply Routing --- 
        if self.routing_layer:
            # Routing layer expects votes [B, InCaps, OutCaps, Pose]
            # and input activations [B, InCaps]
            # Here InCaps is total_in_caps (H*W*InCaps)
            # Corrected: InCaps is total_in_caps_flattened (NumJoints * InCaps)
            poses_out, act_out = self.routing_layer(pose_votes, act_flat, training=training)
        else:
            # Fallback: simple weighted sum if no routing provided
            print("Warning: No routing layer provided to FCQuaternionCaps. Using weighted sum.")
            act_flat_expanded = act_flat[:, :, tf.newaxis, tf.newaxis] # [B, TotalInCaps, 1, 1]
            weighted_votes = pose_votes * act_flat_expanded #[B, TotalInCaps, OutCaps, QuatDims]
            s_j = tf.reduce_sum(weighted_votes, axis=1) # Sum over TotalInCaps -> [B, OutCaps, QuatDims]
            poses_out = tf.nn.l2_normalize(s_j, axis=-1)
            # Calculate activation from norm before normalization
            act_out = tf.norm(s_j, axis=-1) 

        # poses_out: [B, OutCaps, QuatDims]
        # act_out: [B, OutCaps]
        return poses_out, act_out


class QuaternionCapsNet(tf.keras.Model):
    """
    Full QCN combining branched feature extractors for pose & activation,
    primary quaternion caps, intermediate conv quaternion caps, and a final
    fully-connected quaternion caps for camera transformation (4D rot + 3D trans).
    Accepts routing configuration.
    """
    def __init__(self, in_channels=4, out_caps=1, routing_config=None):
        super().__init__()
        
        # Default routing config if none provided
        if routing_config is None:
            routing_config = {"method": "dynamic", "iterations": 3}
        print(f"Initializing QuaternionCapsNet with routing: {routing_config}")

        self.branched_extractor = BranchedResExtractor(
            in_channels_pose=in_channels, 
            in_channels_act=in_channels, 
            mid_channels=32, 
            out_channels=64, 
            stride=1
        )
        self.primary_caps = PrimaryQuatCaps(out_caps=32, quat_dims=4) # Use quat_dims=4
        
        # Instantiate routing layers based on config
        routing_method_name = routing_config.get('method', 'dynamic')
        routing_iters = routing_config.get('iterations', 3)
        routing_kwargs = {k: v for k, v in routing_config.items() if k not in ['method', 'iterations']}

        # Note: ConvCaps routing needs careful spatial handling. 
        # Using simpler FC routing layer here for demonstration.
        # We will apply routing only in the final FC layer.
        # self.conv_caps1 = ConvQuaternionCapsLayer(kernel_size=(1,1), stride=(1,1), in_caps=32, out_caps=32, quat_dims=4) # Removed routing here
        # self.conv_caps2 = ConvQuaternionCapsLayer(kernel_size=(1,1), stride=(1,1), in_caps=32, out_caps=32, quat_dims=4) # Removed routing here
        
        # Using standard Conv2D layers instead of ConvQuaternionCapsLayer for simplicity, 
        # as routing is complex in ConvCaps. Apply routing only in the final FC layer.
        # Replace Conv2D with Conv1D suitable for sequence input (B, NumJoints, QuatDim)
        self.conv1 = tf.keras.layers.Conv1D(filters=128, kernel_size=1, strides=1, padding='valid', activation='relu')
        self.conv2 = tf.keras.layers.Conv1D(filters=32*4, kernel_size=1, strides=1, padding='valid', activation='relu') # Output matches num_primary_caps * quat_dims

        # Final FC Layer with Routing
        fc_routing_layer = get_routing_method(
            name=routing_method_name, 
            out_caps=out_caps, # Final output capsules (e.g., 1 for transform)
            routing_iters=routing_iters, 
            **routing_kwargs
        )
        self.fc_caps = FCQuaternionCaps(
            in_caps=32, # Input capsule types from conv layer
            out_caps=out_caps, 
            quat_dims=4, # Quaternion dimension
            routing_layer=fc_routing_layer
        )

        # Final FC Dense Network for prediction (if needed after capsules)
        # Takes the output capsule pose [B, OutCaps, QuatDims] -> flatten -> Dense(7)
        self.final_flatten = tf.keras.layers.Flatten()
        self.final_fc_output = tf.keras.layers.Dense(7) # 4 for quat, 3 for trans

    def call(self, x, training=False):
        # x expected shape: (B, H, W, in_channels) 
        # Here H=NumJoints, W=1, C=4 (quat dim)
        
        # --- Reshape input for Conv1D --- 
        # Reshape from (B, NumJoints, 1, QuatDim) to (B, NumJoints, QuatDim)
        input_shape = tf.shape(x)
        b = input_shape[0]
        num_joints = input_shape[1]
        quat_dim = input_shape[3]
        x_reshaped = tf.reshape(x, [b, num_joints, quat_dim])

        # --- Apply Conv1D Layers --- 
        features = self.conv1(x_reshaped)
        features = self.conv2(features)
        # Output shape: (B, NumJoints, 32*4)
        
        # --- Reshape features for Primary Caps and FC Routing --- 
        features_shape = tf.shape(features)
        # b = features_shape[0] # Batch size is same
        # num_joints = features_shape[1] # Sequence length is same
        # Channels = features_shape[2] -> should be num_primary_caps * quat_dims
        num_primary_caps = 32 # Must match in_caps for fc_caps
        quat_dims = 4 # Expected by routing
        # Reshape to [B, H*W(NumJoints)*NumCaps, QuatDims] - Needed by FCQuaternionCaps
        # Or adapt FCQuaternionCaps to take [B, NumJoints, NumCaps, QuatDims]
        # Let's adapt FCQuaternionCaps input handling later if needed.
        # For now, reshape assuming features represent the capsules directly along the sequence dim.
        # Treat NumJoints as the spatial dimension H*W for FC Caps
        # Reshape features from (B, NumJoints, NumCaps*QuatDims) -> (B, NumJoints, NumCaps, QuatDims)
        pose_prim = tf.reshape(features, [b, num_joints, num_primary_caps, quat_dims])
        # Simulate activation (e.g., norm of primary capsules)
        act_prim = tf.norm(pose_prim, axis=-1) # Shape: (B, NumJoints, NumCaps)

        # FC quaternion caps => final aggregated quaternion representation
        # Input to fc_caps: pose_prim, act_prim
        pose_fc, act_fc = self.fc_caps(pose_prim, act_prim, training=training)
        # pose_fc shape: [B, OutCaps, QuatDims], act_fc shape: [B, OutCaps]

        # Use the output capsule(s) for prediction
        # If out_caps=1, pose_fc is [B, 1, 4], act_fc is [B, 1]
        # Squeeze the OutCaps dimension if it's 1
        if self.fc_caps.out_caps == 1:
            final_capsule_pose = tf.squeeze(pose_fc, axis=1) # [B, 4]
        else:
            # If multiple output capsules, flatten them or select based on activation
            final_capsule_pose = self.final_flatten(pose_fc) # [B, OutCaps*QuatDims]

        # MLP for final 7D output (Rotation + Translation)
        # Input should be the pose part of the final capsule(s)
        out = self.final_fc_output(final_capsule_pose)
        
        # Split and normalize rotation
        rot = out[:, :4]
        trans = out[:, 4:]
        rot_norm = tf.norm(rot, axis=-1, keepdims=True) + 1e-8
        rot = rot / rot_norm # Ensure unit quaternion output
        
        return tf.concat([rot, trans], axis=-1)
