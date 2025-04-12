import tensorflow as tf
import numpy as np

class AbstractRouting(tf.keras.layers.Layer):
    """Abstract base class for routing methods."""
    def __init__(self, name="AbstractRouting", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, votes, activations_in, **kwargs):
        """
        Args:
            votes: Tensor of votes from lower capsules to higher capsules.
                   Shape: [batch, in_caps_dim1, ..., out_caps, pose_dims]
                   (e.g., [B, H, W, InCaps, OutCaps, Pose] for Conv, [B, InCaps, OutCaps, Pose] for FC)
            activations_in: Tensor of activations for lower capsules.
                           Shape: [batch, in_caps_dim1, ..., in_caps]
        Returns:
            Tuple (poses_out, activations_out) for the higher-level capsules.
            poses_out shape: [batch, out_caps_dim1, ..., out_caps, pose_dims]
            activations_out shape: [batch, out_caps_dim1, ..., out_caps]
        """
        raise NotImplementedError

    def safe_norm(self, s, axis=-1, epsilon=1e-7, keepdims=False):
        # Calculates safe norm, avoiding NaN gradients
        s_norm_sq = tf.reduce_sum(tf.square(s), axis=axis, keepdims=True)
        return tf.sqrt(s_norm_sq + epsilon)


class EMRouting(AbstractRouting):
    """Simplified EM Routing based on QCN principles (Gaussian assumption)."""
    def __init__(self, out_caps, routing_iters=3, beta_v=1.0, beta_a=1.0, epsilon=1e-9, name="EMRouting", **kwargs):
        super().__init__(name=name, **kwargs)
        self.out_caps = out_caps
        self.routing_iters = routing_iters
        # Inverse Temperature parameters (higher value = sharper distribution)
        self.beta_v = beta_v
        self.beta_a = beta_a
        self.epsilon = epsilon # Small value for numerical stability

    def call(self, votes, activations_in, training=False):
        # Input shapes (assuming FC layer input for simplicity):
        # votes: [B, InCaps, OutCaps, Pose(4)]
        # activations_in: [B, InCaps]

        vote_shape = tf.shape(votes)
        batch_size = vote_shape[0]
        in_caps = tf.cast(vote_shape[1], tf.int32)
        pose_dims = vote_shape[3]

        # --- Initializations ---
        # Initialize higher-level capsule activations uniformly
        A_h = tf.fill([batch_size, self.out_caps], 1.0 / tf.cast(self.out_caps, tf.float32))

        # Initialize higher-level poses (Mu_h) using mean of votes initially
        Mu_h = tf.reduce_mean(votes, axis=1) # Shape [B, OutCaps, Pose]
        Mu_h = tf.nn.l2_normalize(Mu_h, axis=-1)

        # --- EM Routing Iterations ---
        for r_iter in range(self.routing_iters):
            # --- E-Step ---
            # Calculate agreement/likelihood

            # Normalize votes (quaternions should be unit)
            votes_norm = tf.nn.l2_normalize(votes, axis=-1) # [B, InCaps, OutCaps, Pose]

            # Cosine similarity (squared dot product for quaternions handles double cover)
            Mu_h_expanded = tf.expand_dims(Mu_h, axis=1) # [B, 1, OutCaps, Pose]
            dot_prod = tf.reduce_sum(votes_norm * Mu_h_expanded, axis=-1) # [B, InCaps, OutCaps]
            # Use abs to handle double cover, then square: (q1 . q2)^2
            cost_v = (1.0 - tf.square(tf.abs(dot_prod))) * self.beta_v # [B, InCaps, OutCaps]

            # Cost for activation (simple difference)
            A_h_expanded = tf.expand_dims(A_h, axis=1) # [B, 1, OutCaps]
            activations_in_expanded = tf.expand_dims(activations_in, axis=-1) # [B, InCaps, 1]
            cost_a = tf.abs(activations_in_expanded - A_h_expanded) * self.beta_a # [B, InCaps, OutCaps]

            # Total cost (negative log likelihood approximation)
            cost_sum = cost_v + cost_a # [B, InCaps, OutCaps]

            # Calculate assignment probabilities R_ih using softmax over output capsules
            log_prior = tf.math.log(A_h_expanded + self.epsilon) # [B, 1, OutCaps]
            log_likelihood = -cost_sum # [B, InCaps, OutCaps]
            log_posterior = log_likelihood + log_prior # [B, InCaps, OutCaps]

            # Softmax over output capsules (axis=2) to get R_ih
            R_ih = tf.nn.softmax(log_posterior, axis=2) # [B, InCaps, OutCaps]

            # --- M-Step ---
            # Update higher-level activations A_h
            R_sum_over_in = tf.reduce_sum(R_ih * activations_in_expanded, axis=1) # [B, OutCaps]
            A_h_total_sum = tf.reduce_sum(R_sum_over_in, axis=1, keepdims=True) + self.epsilon
            A_h = R_sum_over_in / A_h_total_sum
            A_h = tf.clip_by_value(A_h, self.epsilon, 1.0 - self.epsilon) # Keep valid

            # Update higher-level poses Mu_h (weighted average of votes)
            R_ih_expanded = tf.expand_dims(R_ih, axis=-1) # [B, InCaps, OutCaps, 1]
            Mu_h_numerator = tf.reduce_sum(R_ih_expanded * votes, axis=1) # [B, OutCaps, Pose]
            Mu_h_denominator = tf.reduce_sum(R_ih_expanded, axis=1) + self.epsilon # [B, OutCaps, 1]

            # Update Mu_h and normalize (quaternions)
            Mu_h = Mu_h_numerator / Mu_h_denominator
            Mu_h = tf.nn.l2_normalize(Mu_h, axis=-1)

        return Mu_h, A_h # Poses_out, Activations_out


class DynamicRouting(AbstractRouting):
    """Dynamic Routing (Sabour et al., 2017) adapted for Quaternions."""
    def __init__(self, out_caps, routing_iters=3, name="DynamicRouting", **kwargs):
        super().__init__(name=name, **kwargs)
        self.out_caps = out_caps
        self.routing_iters = routing_iters

    def squash(self, s, axis=-1):
        """Squashing activation function for vectors (or quaternions)."""
        s_norm = self.safe_norm(s, axis=axis, keepdims=True)
        scale = s_norm**2 / (1 + s_norm**2) # Original squash scale
        unit_vector = s / s_norm
        return scale * unit_vector

    def call(self, votes, activations_in, training=False):
        # Input shapes (assuming FC layer input):
        # votes (u_hat_ji): [B, InCaps, OutCaps, Pose(4)]
        # activations_in: [B, InCaps] (Not explicitly used in standard DR agreement)

        vote_shape = tf.shape(votes)
        batch_size = vote_shape[0]
        in_caps = tf.cast(vote_shape[1], tf.int32)
        pose_dims = vote_shape[3]

        # Initialize routing logits b_ij to zero
        b_ij = tf.zeros([batch_size, in_caps, self.out_caps], dtype=tf.float32)

        # --- Dynamic Routing Iterations ---\
        v_j = None # Ensure v_j is defined outside the loop
        for r_iter in range(self.routing_iters):
            # Calculate routing weights c_ij = softmax(b_ij) over out_caps
            c_ij = tf.nn.softmax(b_ij, axis=2) # Shape [B, InCaps, OutCaps]

            # Expand c_ij for broadcasting with votes: [B, InCaps, OutCaps, 1]
            c_ij_expanded = tf.expand_dims(c_ij, axis=-1)

            # Calculate weighted sum of votes (higher-level capsule pre-activation s_j)
            # Sum over input capsules (axis=1)
            s_j = tf.reduce_sum(c_ij_expanded * votes, axis=1) # Shape [B, OutCaps, Pose]

            # Apply squashing function to get higher-level capsule output v_j
            v_j = self.squash(s_j, axis=-1) # Shape [B, OutCaps, Pose]

            # Update logits b_ij based on agreement (dot product)
            # Stop gradient for agreement calculation
            v_j_stopped = tf.stop_gradient(v_j)
            # Expand v_j for broadcasting: [B, 1, OutCaps, Pose]
            v_j_expanded = tf.expand_dims(v_j_stopped, axis=1)

            # Agreement (using dot product - normalize?)
            # Normalize votes? Original DR doesn't explicitly normalize votes here
            # but quaternions should ideally be near unit length.
            # Let's try direct dot product first.
            agreement = tf.reduce_sum(votes * v_j_expanded, axis=-1) # Shape [B, InCaps, OutCaps]

            # Update b_ij
            b_ij += agreement # Add agreement to logits

        # Final output pose is v_j
        poses_out = v_j
        # Activation is the norm of the final output pose vector
        activations_out = self.safe_norm(poses_out, axis=-1, keepdims=False) # Remove keepdims

        return poses_out, activations_out


class WeightedSumRouting(AbstractRouting):
    """Simple non-iterative routing using weighted sum based on input activations."""
    def __init__(self, out_caps, name="WeightedSumRouting", **kwargs):
        super().__init__(name=name, **kwargs)
        self.out_caps = out_caps
        self.epsilon = 1e-9

    def call(self, votes, activations_in, training=False):
        # Input shapes (assuming FC layer input):
        # votes: [B, InCaps, OutCaps, Pose(4)]
        # activations_in: [B, InCaps]

        # Expand activations_in for weighting votes: [B, InCaps, 1, 1]
        a_in_expanded = activations_in[:, :, tf.newaxis, tf.newaxis]

        # Weight votes by input activations
        weighted_votes = votes * a_in_expanded # Shape [B, InCaps, OutCaps, Pose]

        # Sum weighted votes over input capsules
        s_j = tf.reduce_sum(weighted_votes, axis=1) # Shape [B, OutCaps, Pose]

        # Normalize pose (quaternion)
        poses_out = tf.nn.l2_normalize(s_j, axis=-1)

        # Calculate activation as norm of summed vector before normalization
        activations_out = self.safe_norm(s_j, axis=-1, keepdims=False) # Remove keepdims

        return poses_out, activations_out


# Factory function to get routing method by name
def get_routing_method(name, out_caps, routing_iters=3, **kwargs):
    """
    Factory function to instantiate a routing layer.

    Args:
        name (str): Name of the routing method ('em', 'dynamic', 'weighted_sum').
        out_caps (int): Number of output capsules for the layer using this routing.
        routing_iters (int): Number of iterations for iterative methods (EM, Dynamic).
        **kwargs: Additional keyword arguments specific to the routing method (e.g., beta_v for EM).

    Returns:
        An instance of the specified routing layer.
    """
    name_lower = name.lower() if name else 'dynamic' # Default to dynamic if None or empty

    if name_lower == 'em':
        print(f"Using EM Routing (iters={routing_iters})")
        # Extract EM specific kwargs or use defaults
        beta_v = kwargs.get('beta_v', 1.0)
        beta_a = kwargs.get('beta_a', 1.0)
        epsilon = kwargs.get('epsilon', 1e-9)
        return EMRouting(out_caps=out_caps, routing_iters=routing_iters,
                         beta_v=beta_v, beta_a=beta_a, epsilon=epsilon)
    elif name_lower == 'dynamic':
        print(f"Using Dynamic Routing (iters={routing_iters})")
        return DynamicRouting(out_caps=out_caps, routing_iters=routing_iters)
    elif name_lower == 'weighted_sum':
        print("Using Weighted Sum Routing")
        return WeightedSumRouting(out_caps=out_caps)
    else:
        raise ValueError(f"Unknown routing method: '{name}'. Choose 'em', 'dynamic', or 'weighted_sum'.")
