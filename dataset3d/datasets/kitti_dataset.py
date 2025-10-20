import os
import numpy as np
import torch
from typing import Any, Dict, List, Optional
from dataset3d.datasets.base_dataset import Base3DDataset
from dataset3d.core.geometry import transform_bbox, get_transform_between_frames
from dataset3d.utils.io import load_bin_pointcloud
from dataset3d.core.unified_format import BBox3D, PointCloud, CalibrationData, Frame, Sample


class KITTIDataset(Base3DDataset):
    """KITTI 3D Object Detection Dataset
    
    Coordinate Frame Conventions:
    - Default frame: "camera" (rectified camera 0)
        * X: Right, Y: Down, Z: Forward
        * Boxes and labels are in camera frame
    - LiDAR frame: "lidar" (Velodyne)
        * X: Forward, Y: Left, Z: Up
        * Point clouds in LiDAR frame
    - Calibration provides: camera intrinsics, lidar_to_camera transforms
    
    Dataset-specific notes:
    - KITTI ground truth boxes are in camera coordinate frame
    - Can optionally transform to LiDAR frame using target_frame parameter
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        modalities: List[str] = ["lidar"],
        use_sequences: bool = False,
        sequence_length: int = 5,
        target_frame: str = "camera",
        **kwargs
    ):
        """Initialize KITTI dataset with configurable modalities
        
        Args:
            target_frame: Coordinate frame for output data
                - "camera": Keep boxes in camera frame (KITTI native)
                - "lidar": Transform boxes to LiDAR frame
        """
        self.use_sequences = use_sequences
        self.sequence_length = sequence_length
        self.target_frame = target_frame
        super().__init__(data_root=data_root, split=split, modalities=modalities, target_frame=target_frame, **kwargs)

    # -------------------------------------------------------------------------
    def _load_sample_list(self) -> List[str]:
        """Read frame IDs from ImageSets/{split}.txt"""
        list_path = os.path.join(self.data_root, "ImageSets", f"{self.split}.txt")
        with open(list_path, "r") as f:
            frame_ids = [line.strip() for line in f.readlines() if line.strip()]
        return frame_ids

    # -------------------------------------------------------------------------
    def load_raw_data(self, idx: int) -> Dict[str, Any]:
        """Load velodyne points, labels, and calibration"""
        frame_id = self.sample_list[idx]
        split_dir = "training" if self.split != "test" else "testing"

        # --- Load point cloud (.bin)
        lidar_path = os.path.join(self.data_root, split_dir, "velodyne", f"{frame_id}.bin")
        lidar_points = load_bin_pointcloud(lidar_path)

        # --- Load calibration
        calib_path = os.path.join(self.data_root, split_dir, "calib", f"{frame_id}.txt")
        calib_data = self._parse_calibration(calib_path)

        # --- Load labels 
        label_path = os.path.join(self.data_root, split_dir, "label_2", f"{frame_id}.txt")
        labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split(" ")
                    cls_name = parts[0]
                    if cls_name.lower() == "dontcare":
                        continue
                    h, w, l = map(float, parts[8:11])
                    x, y, z = map(float, parts[11:14])
                    yaw = float(parts[14])
                    labels.append({
                        "class_name": cls_name,
                        "dimensions": (l, w, h),
                        "center": (x, y, z),
                        "rotation_yaw": yaw,
                    })

        return {
            "frame_id": frame_id,
            "lidar_points": lidar_points,
            "labels": labels,
            "calibration": calib_data,
        }

    # -------------------------------------------------------------------------
    def _parse_calibration(self, calib_path: str) -> CalibrationData:
        """Parse KITTI calibration file"""
        with open(calib_path, "r") as f:
            lines = f.readlines()

        P2 = np.array([float(x) for x in lines[2].split()[1:]]).reshape(3, 4)
        Tr_velo_to_cam = np.array([float(x) for x in lines[5].split()[1:]]).reshape(3, 4)

        T_lidar_to_cam = np.eye(4)
        T_lidar_to_cam[:3, :4] = Tr_velo_to_cam

        calib = CalibrationData(
            camera_intrinsic={"cam2": torch.tensor(P2[:, :3], dtype=torch.float32)},
            lidar_to_camera={"cam2": torch.tensor(T_lidar_to_cam, dtype=torch.float32)},
        )
        return calib

    # -------------------------------------------------------------------------
    def to_unified(self, raw_data: Dict[str, Any]) -> Sample:
        """Parse KITTI format to unified Sample"""
        frame_id = raw_data["frame_id"]
        lidar_points = torch.tensor(raw_data["lidar_points"], dtype=torch.float32)
        calib = raw_data["calibration"]

        # Create PointCloud
        point_cloud = PointCloud(
            points=lidar_points[:, :3],
            coordinate_frame="lidar",
            sensor_id="lidar_top",
        )

        # Create BBoxes
        bboxes = []
        for obj in raw_data["labels"]:
            unified_name, unified_id = self.class_registry.map_dataset_class_to_unified("kitti", obj["class_name"])
            bbox = BBox3D(
                center=torch.tensor(obj["center"], dtype=torch.float32),
                dimensions=torch.tensor(obj["dimensions"], dtype=torch.float32),
                rotation_yaw=obj["rotation_yaw"],
                coordinate_frame="camera",
                class_name=unified_name,
                class_id=unified_id,
            )
            bboxes.append(bbox)

        # Transform to target frame if needed
        if self.target_frame == "lidar":
            T_cam_to_lidar = torch.inverse(calib.lidar_to_camera["cam2"])
            bboxes = [transform_bbox(b, T_cam_to_lidar, target_frame="lidar") for b in bboxes]

        frame = Frame(
            frame_id=frame_id,
            timestamp=0.0,
            point_cloud=point_cloud,
            bboxes_3d=bboxes,
            calibration=calib,
            has_labels=len(bboxes) > 0,
        )

        return Sample(data=frame, split=self.split, dataset_name="kitti")
