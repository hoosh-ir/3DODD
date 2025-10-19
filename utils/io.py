import numpy as np 
def load_bin_pointcloud(file_path: str) -> np.ndarray:
    """Load KITTI-style .bin point cloud file
    
    Each .bin file contains consecutive float32 values representing:
        [x, y, z, reflectance]
    There are 4 floats per point, so the data length must be divisible by 4.

    Args:
        file_path (str): Path to .bin point cloud file

    Returns:
        np.ndarray: Array of shape (N, 4) where each row is [x, y, z, intensity]
    """
    # Read binary file as float32
    point_cloud = np.fromfile(file_path, dtype=np.float32)

    # Ensure correct shape (N, 4)
    point_cloud = point_cloud.reshape(-1, 4)

    return point_cloud

# def load_pcd_pointcloud(file_path: str) -> np.ndarray:
#     """Load .pcd format point cloud file"""

# def save_to_json(data: Sample, file_path: str) -> None:
#     """Serialize Sample to JSON with tensor encoding"""

# def load_from_json(file_path: str) -> Sample:
#     """Deserialize Sample from JSON"""