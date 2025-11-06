import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import rerun as rr
import torch
import numpy as np
from dataset3d.core.unified_format import Sample, Sequence, Frame
from dataset3d.visualization.pointcloud_vis import visualize_pointcloud, visualize_boxes_3d
from dataset3d.visualization.image_vis import visualize_image
from dataset3d.visualization.bev_vis import visualize_bev

class RerunVisualizer:
    def __init__(self, recording_name: str = "3dodd"):
        rr.init(recording_name, spawn=True)

    def log_sample(
        self,
        sample: Sample,
        entity_path: str = "/world/sequence",
        color_mode: str = "intensity"
    ) -> None:
        """Log one frame (point cloud + boxes) under a consistent path, replacing old ones."""
        data = sample.data
        if not isinstance(data, Frame):
            raise TypeError("RerunVisualizer currently supports only single-agent Frame visualization.")

        # Coordinate system (rerun ignores duplicates)
        rr.log("world", rr.ViewCoordinates.RFU , static= True)

        # 💡 Clear previous logs before writing new frame
        rr.log(entity_path, rr.Clear(recursive=True))

        # Compute centroid (optional)
        pts = data.point_cloud.points[:, :3].cpu().numpy()
        centroid = pts.mean(axis=0)

        # Log point cloud — single persistent entity
        visualize_pointcloud(
            data.point_cloud.points,
            color_by=color_mode,
            entity_path=f"{entity_path}/pointcloud"
        )



        # Log bounding boxes — single persistent entity
        if hasattr(data, "bboxes_3d") and len(data.bboxes_3d) > 0:
            visualize_boxes_3d(
                data.bboxes_3d,
                entity_path=f"{entity_path}/bboxes"
            )

        # Optionally log corresponding camera image
        if hasattr(data, "frame_id"):
            img_path = os.path.join(
                "/media/sina/New Volume/data/kitti/training/image_2",
                f"{int(data.frame_id):06d}.png"
            )

            visualize_image(img_path, entity_path=f"{entity_path}/image")

        

        # Log current frame ID text
        rr.log("frame_id", rr.TextLog(f"Frame: {data.frame_id}"))
        print("Frame id:", data.frame_id)
        print("Expected file:", f"{data.frame_id}.png")


    def log_sequence(
        self,
        sequence: Sequence,
        timeline: str = "frame"
    ) -> None:
        """Not implemented for non-sequential datasets like KITTI."""
        raise NotImplementedError(
            "log_sequence() is not implemented yet. "
            "This method is only intended for sequential datasets "
            "(e.g., nuScenes, Waymo, or OPV2V). "
            "For KITTI, use log_sample() or show_dataset() instead."
        )
