import numpy as np
import pyquaternion as pq
import tensorflow as tf
from dataloader import PoseDataLoader
from utils import get_parent_map , view_matrix_to_quaternion , compute_joint_quaternions_3d , compute_joint_quaternions_2d

# -----------------------------------------------------------------------------
# QCNDataloader
# -----------------------------------------------------------------------------

class QCNDataloader(PoseDataLoader):
    """
    Dataloader for Quaternion Capsule Networks.
    
    It converts the loaded data into quaternion representations:
      - kp3d_quat: a (31, 4) array of per-joint quaternions in world space.
      - kp2d_quat: a (31, 4) array of per-joint quaternions in camera space.
      - view_rot_quat: the view matrix rotation (as a 4-element quaternion).
      - view_trans: the translation vector extracted from the view matrix.
      
    The input to the model will be the kp2d_quat and the target output will be
    the view rotation and translation.
    """
    def __init__(self, train_file, val_file, projection_matrix=None):
        super().__init__(train_file, val_file)
        self.parent_map = get_parent_map()
        self.projection_matrix = projection_matrix  # Store the (4, 4) matrix
    
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        
        # Data from the base loader:
        vm = sample['vm']               # 4x4 view matrix
        kp2d = sample['kp2d_camera']      # (31, 2) array (in camera space)
        kp3d = sample['kp3d_world']       # (31, 3) array (in world space)
        
        # --- Extract view matrix rotation and translation ---
        # Note: In our dataset, the translation is typically in vm[:3, 3]
        # If your view matrix has translation elsewhere, adjust accordingly.
        view_rot_quat = view_matrix_to_quaternion(vm)
        # Assuming translation is in the standard position
        view_trans = vm[:3, 3] 
        
        # --- Convert keypoints to per-joint quaternions ---
        # For 3D, we use a reference vector [0,1,0] (Y is up).
        kp3d_quat = compute_joint_quaternions_3d(kp3d, self.joint_order, self.parent_map, ref=np.array([0, 1, 0]))
        # For 2D, we use a reference vector [1,0] (i.e. 0° is along +X) and rotation about Z.
        kp2d_quat = compute_joint_quaternions_2d(kp2d, self.joint_order, self.parent_map, ref=np.array([1, 0]))
        
        return {
            'kp3d_quat': kp3d_quat,                   # (31, 4) array
            'kp2d_quat': kp2d_quat,                   # (31, 4) array
            'view_rot_quat': view_rot_quat,           # (4,) array [w, x, y, z]
            'view_trans': view_trans,                 # (3,) array
            'kp2d_gt': kp2d,                        # (31, 2) array
            'kp3d_world': kp3d                      # (31, 3) array
        }
    
    def prepare_data(self, batch_size, is_training=True):
        data_source = self.train_data if is_training else self.val_data
        
        def generator():
            indices = list(range(len(data_source)))
            if is_training:
                np.random.shuffle(indices)
            for idx in indices:
                sample = self.__getitem__(idx)
                yield (
                    sample['kp2d_quat'],     # Input: (31, 4) 2D joint quaternions
                    sample['view_rot_quat'], # Target: (4,) view rotation quaternion
                    sample['view_trans'],    # Target: (3,) view translation vector
                    sample['kp2d_gt'],       # Ground Truth: (31, 2) 2D keypoints
                    sample['kp3d_world'],    # Ground Truth: (31, 3) 3D keypoints
                    self.projection_matrix # Yield projection matrix
                )
        
        output_signature = (
            tf.TensorSpec(shape=(len(self.joint_order), 4), dtype=tf.float32),  # kp2d_quat
            tf.TensorSpec(shape=(4,), dtype=tf.float32),                        # view_rot_quat
            tf.TensorSpec(shape=(3,), dtype=tf.float32),                        # view_trans
            tf.TensorSpec(shape=(len(self.joint_order), 2), dtype=tf.float32),  # kp2d_gt
            tf.TensorSpec(shape=(len(self.joint_order), 3), dtype=tf.float32),  # kp3d_world
            tf.TensorSpec(shape=(4, 4), dtype=tf.float32)                      # projection_matrix
        )
        
        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
        if is_training:
            dataset = dataset.shuffle(buffer_size=min(len(data_source), 1000))
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

# -----------------------------------------------------------------------------
# Example usage:
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    data_path  = './data/train_data_20.json'
    val_data   = './data/val_data_20.json'
    
    # (Optionally) define the projection matrix if needed.
    projection_matrix = np.array([2.790552,  0.,         0.,         0.,
                                  0.,         1.5696855,  0.,         0.,
                                  0.,         0.,        -1.0001999, -1.,
                                  0.,         0.,        -0.20002,    0.], dtype=np.float32)
    projection_matrix = projection_matrix.reshape((4, 4))
    
    qc_loader = QCNDataloader(data_path, val_data, projection_matrix=projection_matrix)
    sample_qcn = qc_loader.__getitem__(1)
    
    print("KP3D Quaternions shape:", sample_qcn['kp3d_quat'].shape)  # Expect (31, 4)
    print("KP2D Quaternions shape:", sample_qcn['kp2d_quat'].shape)  # Expect (31, 4)
    print("View Rotation Quaternion:", sample_qcn['view_rot_quat'])
    print("View Translation:", sample_qcn['view_trans'])

    batch_size = 32 
    train_loader = qc_loader.prepare_data(batch_size,is_training=True)
    # Example of getting data from the loader
    for kp2d_q, view_r_q, view_t, kp2d, kp3d, proj_mat in train_loader.take(1):
        print("\nLoaded batch shapes:")
        print("kp2d_quat shape:", kp2d_q.shape)
        print("view_rot_quat shape:", view_r_q.shape)
        print("view_trans shape:", view_t.shape)
        print("kp2d_gt shape:", kp2d.shape)
        print("kp3d_world shape:", kp3d.shape)
        print("projection_matrix shape:", proj_mat.shape)
    # print("\nLoaded the data using prepare_data ") # Redundant print