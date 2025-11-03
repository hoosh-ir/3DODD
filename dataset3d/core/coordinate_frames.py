from typing import Dict

class CoordinateFrame:
    """Coordinate frame definitions and conventions
    
    Supported coordinate frames with their conventions:
    
    1. "lidar": LiDAR sensor frame
       - Origin: LiDAR sensor position
       - Common conventions:
         * X: Forward, Y: Left, Z: Up (most common: Waymo, nuScenes)
         * X: Forward, Y: Right, Z: Down (some systems)
       - Use dataset metadata to specify exact convention
    
    2. "camera": Camera sensor frame (rectified)
       - Origin: Camera optical center
       - Common conventions:
         * X: Right, Y: Down, Z: Forward (KITTI, OpenCV standard)
         * X: Right, Y: Up, Z: Forward (some systems)
       - Use dataset metadata to specify exact convention
    
    3. "ego": Ego vehicle frame
       - Origin: Vehicle center (typically center of rear axle or vehicle center)
       - X: Forward, Y: Left, Z: Up (right-handed, most common)
    
    4. "global": Global/world frame
       - Origin: Map/world origin
       - Typically: X: East, Y: North, Z: Up (ENU convention)
       - Or: X: North, Y: East, Z: Down (NED convention)
    
    Each dataset loader should document its coordinate frame conventions.
    """
    
    VALID_FRAMES = ["lidar", "camera", "ego", "global"]
    
    # Dataset-specific coordinate conventions (axis directions)
    DATASET_CONVENTIONS = {
        "kitti": {
            "camera": {"x": "right", "y": "down", "z": "forward"},  # Rectified camera
            "lidar": {"x": "forward", "y": "left", "z": "up"}       # Velodyne
        },
        "nuscenes": {
            "lidar": {"x": "forward", "y": "left", "z": "up"},
            "camera": {"x": "right", "y": "down", "z": "forward"},
            "ego": {"x": "forward", "y": "left", "z": "up"},
            "global": {"x": "east", "y": "north", "z": "up"}  # ENU
        },
        "waymo": {
            "lidar": {"x": "forward", "y": "left", "z": "up"},
            "camera": {"x": "right", "y": "down", "z": "forward"},
            "ego": {"x": "forward", "y": "left", "z": "up"}
        },
        # Add other datasets...
    }
    
    @staticmethod
    def validate_frame(frame: str) -> bool:
        """Validate coordinate frame name"""
        return frame in CoordinateFrame.VALID_FRAMES
    
    @staticmethod
    def get_convention(dataset_name: str, frame: str) -> Dict[str, str]:
        """Get axis convention for dataset and frame
        
        Returns:
            Dict with 'x', 'y', 'z' keys mapping to direction strings
        """
        dataset_name = dataset_name.lower()
        frame = frame.lower()

        if dataset_name not in CoordinateFrame.DATASET_CONVENTIONS:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Supported datasets: {list(CoordinateFrame.DATASET_CONVENTIONS.keys())}"
            )

        dataset_frames = CoordinateFrame.DATASET_CONVENTIONS[dataset_name]
        if frame not in dataset_frames:
            raise ValueError(
                f"Frame '{frame}' not defined for dataset '{dataset_name}'. "
                f"Available frames: {list(dataset_frames.keys())}"
            )

        return dataset_frames[frame]
