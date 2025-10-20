# core/unified_format.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List, Union

import torch


@dataclass
class BBox3D:
    """3D Bounding Box with explicit coordinate frame specification

    Each bbox explicitly specifies its coordinate frame to support different dataset conventions.
    Use coordinate_frame to know which system this bbox is defined in.
    """
    # ---- required (non-default) fields must come first ----
    center: torch.Tensor            # (3,) xyz in meters
    dimensions: torch.Tensor        # (3,) length, width, height in meters
    rotation_yaw: float             # Yaw angle in radians, rotation around Z-axis of coordinate_frame
    coordinate_frame: str           # "lidar", "camera", "ego", "global" - explicit frame specification
    class_name: str                 # Human-readable class name (e.g., "Car", "Pedestrian")
    class_id: int                   # Dataset-specific class ID (use ClassRegistry for unified mapping)

    # ---- optional (default) fields come after ----
    rotation_quat: Optional[torch.Tensor] = None  # (4,) [w,x,y,z] for full 3D rotation (if available)
    confidence: float = 1.0                       # Confidence score [0, 1], 1.0 for ground truth
    tracking_id: Optional[int] = None             # Unique ID for multi-object tracking across frames
    velocity: Optional[torch.Tensor] = None       # (3,) vx,vy,vz in m/s in same coordinate_frame
    attributes: Dict[str, Any] = field(default_factory=dict)  # Dataset-specific attributes


# Future: 3D Segmentation Support
# @dataclass
# class SegmentationMask3D:
#     """3D segmentation mask for point clouds"""
#     points: torch.Tensor  # (N, 3) xyz coordinates
#     labels: torch.Tensor  # (N,) per-point class labels
#     instance_ids: Optional[torch.Tensor] = None  # (N,) per-point instance IDs
#     semantic_labels: Optional[torch.Tensor] = None  # (N,) semantic segmentation
#     panoptic_labels: Optional[torch.Tensor] = None  # (N,) panoptic segmentation


@dataclass
class PointCloud:
    """Point cloud data with explicit coordinate frame specification"""
    points: torch.Tensor               # (N, 3+) xyz + optional (intensity, ring, timestamp, ...), dtype=float32
    coordinate_frame: str              # "lidar", "camera", "ego", "global" - explicit frame specification
    sensor_id: str = "lidar"           # Sensor identifier (e.g., "lidar_top", "lidar_front")
    timestamp: float = 0.0             # Unix timestamp in seconds


@dataclass
class CalibrationData:
    """Sensor calibration and extrinsic parameters

    Transformation Matrix Convention: All 4x4 matrices are in SE(3) format
    Apply as: points_target = (T @ points_source.T).T[:, :3] for (N,3) points

    Dataset Availability:
    - KITTI: camera_intrinsic, lidar_to_camera (no ego frame)
    - nuScenes: all fields available
    - Waymo: all fields available
    - DAIR-V2X: depends on sensor setup
    - OPV2V: lidar_to_ego, camera_to_ego
    """
    camera_intrinsic: Optional[Dict[str, torch.Tensor]] = None  # {camera_name: (3,3) K matrix}
    camera_distortion: Optional[Dict[str, torch.Tensor]] = None  # {camera_name: (5,) or (8,) coeffs}
    lidar_to_camera: Optional[Dict[str, torch.Tensor]] = None   # {camera_name: (4,4) transform}
    camera_to_ego: Optional[Dict[str, torch.Tensor]] = None     # {camera_name: (4,4) transform}
    lidar_to_ego: Optional[torch.Tensor] = None                 # (4,4) transform, often identity for ego-frame datasets


@dataclass
class Frame:
    """Single sensor frame from one agent at one timestamp

    Coordinate System: Specified per-element via coordinate_frame field.
    - Point cloud has coordinate_frame attribute
    - Each bbox has coordinate_frame attribute
    - May be heterogeneous (e.g., points in lidar frame, boxes in camera frame)
    - Use utils/geometry.py to transform between frames
    """
    frame_id: str                            # Unique frame identifier
    timestamp: float                         # Unix timestamp in seconds
    point_cloud: PointCloud                  # Point cloud data (with its coordinate_frame)
    bboxes_3d: List[BBox3D]                  # Ground truth boxes (with their coordinate_frame), empty during inference
    calibration: CalibrationData             # Sensor calibration for transformations
    images: Optional[Dict[str, torch.Tensor]] = None  # {camera_name: (H,W,3) uint8 RGB}, lazy-loaded
    agent_id: Optional[str] = None                    # Agent identifier for cooperative scenarios
    ego_pose: Optional[torch.Tensor] = None           # (4,4) global pose matrix in world frame
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional dataset-specific data
    has_labels: bool = True                   # False during inference when ground truth unavailable
    # Future: segmentation_masks: Optional[List[SegmentationMask3D]] = None


@dataclass
class CooperativeFrame:
    """Multi-agent frame with data from multiple agents

    Coordinate System: Each agent's Frame is in their local frame.
    Use ego_pose to transform to the global frame, or use cooperative/transforms.py utilities.

    Temporal Alignment: Agents may have slightly misaligned timestamps (< 100ms typical).
    Use timestamp_tolerance to verify synchronization quality.
    """
    frame_id: str                                          # Unique cooperative frame identifier
    timestamp: float                                       # Reference timestamp (typically from primary/ego agent)
    agents: Dict[str, Frame]                               # {agent_id: Frame} - use dict for explicit agent identification
    timestamp_tolerance: float = 0.1                       # Max timestamp difference between agents (seconds)
    available_agents: List[str] = field(default_factory=list)  # List of agent IDs present
    missing_agents: List[str] = field(default_factory=list)    # Expected but missing agents


@dataclass
class Sequence:
    """Temporal sequence of frames

    Use for temporal/sequential tasks like tracking, prediction, or temporal detection.
    Sequence must be homogeneous: either all Frame or all CooperativeFrame.
    """
    sequence_id: str                         # Unique sequence identifier
    frames: List[Frame]                      # Single-agent sequence (use CooperativeSequence for multi-agent)
    frame_indices: List[int]                 # Maps to global dataset frame indices
    timestamps: List[float]                  # Timestamp for each frame (must be monotonically increasing)
    is_continuous: bool = True               # False if there are temporal gaps in the sequence


@dataclass
class CooperativeSequence:
    """Temporal sequence of cooperative frames (multi-agent)"""
    sequence_id: str
    frames: List[CooperativeFrame]
    frame_indices: List[int]
    timestamps: List[float]
    is_continuous: bool = True
    consistent_agents: List[str] = field(default_factory=list)  # Agents present in all frames


@dataclass
class Sample:
    """Single sample: Frame, CooperativeFrame, Sequence, or CooperativeSequence"""
    data: Union[Frame, CooperativeFrame, Sequence, CooperativeSequence]
    split: str                         # "train", "val", "test"
    dataset_name: str = ""             # Source dataset name (e.g., "kitti", "nuscenes")

    def is_single_frame(self) -> bool:
        """Check if sample is a single frame (not sequence)"""
        return isinstance(self.data, (Frame, CooperativeFrame))

    def is_cooperative(self) -> bool:
        """Check if sample contains multi-agent data"""
        return isinstance(self.data, (CooperativeFrame, CooperativeSequence))

    def is_sequence(self) -> bool:
        """Check if sample is a temporal sequence"""
        return isinstance(self.data, (Sequence, CooperativeSequence))
