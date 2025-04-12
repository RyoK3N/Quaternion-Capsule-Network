#Import necessary libraries
import numpy as np 
import tensorflow as tf 
import json
import logging
import os

#Set up logging
logger = logging.getLogger(__name__)

#Define the PoseDataLoader class
class PoseDataLoader():
    """
    A dataloader for VMP2D.
    Loads the data from the json file and returns a dataset.
    Class Methods:
    - __init__: Initialize the dataloader.
    - load_data: Load the data from the json file.
    - __len__: Return the length of the dataset.
    - __getitem__: Return the data at the given index.
    - flatten_keypoints: Flatten the keypoints dictionary into a 1D array.
    - normalize_keypoints: Normalize the keypoints.
    """
    def __init__(self, train_file, val_file):
        """
        Initialize the dataloader with the training and validation data.
        Args:
            train_file: The path to the training data file.
            val_file: The path to the validation data file.
        """
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.joint_order = [
            'Chest', 'Head', 'HeadEnd', 'Hips', 'LeftArm', 'LeftFinger',
            'LeftFingerEnd', 'LeftFoot', 'LeftForearm', 'LeftHand', 'LeftHeel',
            'LeftLeg', 'LeftShoulder', 'LeftThigh', 'LeftToe', 'LeftToeEnd',
            'Neck', 'RightArm', 'RightFinger', 'RightFingerEnd', 'RightFoot',
            'RightForearm', 'RightHand', 'RightHeel', 'RightLeg', 'RightShoulder',
            'RightThigh', 'RightToe', 'RightToeEnd', 'SpineLow', 'SpineMid'
        ]
        self.train_data = self.load_data(train_file)
        self.val_data = self.load_data(val_file)
              
    def load_data(self, file_path):
        """
        Load the data from the json file.
        Args:
            file_path: The path to the data file.
        Returns:
            The data from the json file.
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            logging.info(f"Loaded {len(data)} samples from {file_path}")
            return data
        except Exception as e:
            logging.error(f"Error loading data from {file_path}: {e}")
            return []

    def __len__(self):
        """
        Return the length of the dataset.
        Returns:
            The length of the dataset.
        """
        return len(self.train_data)
    
    def __getitem__(self, idx):
        """
        Get the data at the given index.
        Args:
            idx: The index of the data.
        Returns:
            The data at the given index.
        """
        data_point = self.train_data[idx]
        kpts_2d = self.flatten_keypoints(data_point['kps_2d'])
        kpts_3d = self.flatten_keypoints(data_point['kps_3d'])
        normalized_kpts = self.normalize_keypoints(kpts_2d)
        view_matrix = self.process_camera_matrix(data_point['camera_matrix'])
        
        # Normalize translation components
        normalized_view_matrix = view_matrix.copy()
        rotation_min = np.min(view_matrix[:3, :3])
        rotation_max = np.max(view_matrix[:3, :3])
        rotation_range = rotation_max - rotation_min
        
        # Scale down large translation values to be in the same range as rotation values
        translation_scale_factor = 1.0 / (np.max(np.abs(view_matrix[:3, 3])) + 1e-8)
        normalized_view_matrix[:3, 3] = view_matrix[:3, 3] * translation_scale_factor * rotation_range
     
        return {
            'kp_gt_flat': normalized_kpts,
            'vm_norm_flat': normalized_view_matrix,
            'vm': view_matrix,
            'kp2d_camera': kpts_2d.reshape(-1, 2),
            'kp3d_world': kpts_3d.reshape(-1, 3)
        }
    
    def flatten_keypoints(self, keypoints_dict):
        """
        Flattens the keypoints dictionary into a 1D array following the joint order.
        Skips keys that aren't in the joint order (like 'date' or 'Body').
        """
        arr = []
        for joint in self.joint_order:
            if joint in keypoints_dict and joint not in ['date', 'Body']:
                arr.extend(keypoints_dict[joint])
        return np.array(arr, dtype=np.float64)
    
    def normalize_keypoints(self, keypoints):
        """
        Normalize the keypoints to improve training stability.
        This is a simple mean centering and scaling operation.
        """
        # dim = 2 if len(keypoints) % 2 == 0 else 3
        # num_joints = len(keypoints) // dim
        # keypoints_reshaped = keypoints.reshape(num_joints, dim)
        # center = np.mean(keypoints_reshaped, axis=0)
        # centered = keypoints_reshaped - center
        # distances = np.linalg.norm(centered, axis=1)
        # scale = np.max(distances) if np.max(distances) > 0 else 1.0
        # normalized = centered / (scale + 1e-8)
        # return normalized.reshape(-1)
        return keypoints
        
    
    def process_camera_matrix(self, camera_matrix):
        """
        Process the camera matrix into the required 4x4 format.
        Treating it as a view matrix (world to camera transformation).
        """
        matrix = np.eye(4, dtype=np.float64)
        if isinstance(camera_matrix, list):
            if len(camera_matrix) == 16:
                matrix = np.array(camera_matrix, dtype=np.float64).reshape(4, 4)
            elif len(camera_matrix) == 12:
                matrix[:3, :4] = np.array(camera_matrix, dtype=np.float64).reshape(3, 4)
            elif len(camera_matrix) < 16:
                rotation = np.array(camera_matrix[:9], dtype=np.float64).reshape(3, 3)
                translation = np.array(camera_matrix[9:], dtype=np.float64)
                matrix[:3, :3] = rotation
                matrix[:3, 3] = translation
        elif isinstance(camera_matrix, np.ndarray):
            if camera_matrix.shape == (3, 4):
                matrix[:3, :4] = camera_matrix.astype(np.float64)
            elif camera_matrix.shape == (4, 4):
                matrix = camera_matrix.astype(np.float64)
        else:
            matrix = np.array(camera_matrix, dtype=np.float64).reshape(4, 4)
        
        return matrix
    
    def EDA(self,):
        """
        Function for Exploring the dataset.
        Finds Min-Max of 2dKps in the datset
        Finds Min-Max of 3dKps in the datset
        Finds Min-Max of camera_matrix in the datset

        """
    
    def prepare_data(self, batch_size, is_training=True):
        """
        Prepare batched datasets for training or validation.
        """
        data_source = self.train_data if is_training else self.val_data

        def generator():
            indices = list(range(len(data_source)))
            if is_training:
                np.random.shuffle(indices)

            for idx in indices:
                sample = data_source[idx]
                kpts_2d = self.flatten_keypoints(sample['kps_2d'])
                kpts_3d = self.flatten_keypoints(sample['kps_3d'])
                normalized_kpts = self.normalize_keypoints(kpts_2d)
                view_matrix = self.process_camera_matrix(sample['camera_matrix'])

                # Normalize translation components
                normalized_view_matrix = view_matrix.copy()
                rotation_min = np.min(view_matrix[:3, :3])
                rotation_max = np.max(view_matrix[:3, :3])
                rotation_range = rotation_max - rotation_min

                # Scale down large translation values to be in the same range as rotation values
                translation_scale_factor = 1.0 / (np.max(np.abs(view_matrix[:3, 3])) + 1e-8)
                normalized_view_matrix[:3, 3] = view_matrix[:3, 3] * translation_scale_factor * rotation_range
                kpts_2d_reshaped = kpts_2d.reshape(-1, 2)
                kpts_3d_reshaped = kpts_3d.reshape(-1, 3)

                yield (
                    normalized_kpts,
                    normalized_view_matrix,
                    view_matrix,  
                    kpts_2d_reshaped,
                    kpts_3d_reshaped
                )

        num_joints = len(self.joint_order)
        input_dim = num_joints * 2

        # Combine type and shape information using output_signature
        output_signature = (
            tf.TensorSpec(shape=[input_dim], dtype=tf.float64),   # normalized_kpts
            tf.TensorSpec(shape=[4, 4], dtype=tf.float64),          # normalized_view_matrix
            tf.TensorSpec(shape=[4, 4], dtype=tf.float64),          # original_view_matrix
            tf.TensorSpec(shape=[num_joints, 2], dtype=tf.float64),   # keypoints_2d
            tf.TensorSpec(shape=[num_joints, 3], dtype=tf.float64)    # keypoints_3d
        )

        # Create the dataset using only output_signature
        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=output_signature
        )

        if is_training:
            dataset = dataset.shuffle(buffer_size=min(len(data_source), 1000))

        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
