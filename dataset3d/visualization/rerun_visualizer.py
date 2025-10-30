import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import rerun as rr
import torch
import numpy as np
from dataset3d.core.unified_format import Sample, Sequence, Frame
from dataset3d.visualization.pointcloud_vis import visualize_pointcloud, visualize_boxes_3d


class RerunVisualizer:
    def __init__(self, recording_name: str = "3dodd"):
        rr.init(recording_name, spawn=True)

    def log_sample(
        self,
        sample: Sample,
        entity_path: str = "/",
        color_mode: str = "intensity"
    ) -> None:
        data = sample.data
        if not isinstance(data, Frame):
            raise TypeError("RerunVisualizer currently supports only single-agent Frame visualization.")

        # Set up coordinate system
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP)

        # Set up camera with good view of the scene
        pts = data.point_cloud.points[:, :3].cpu().numpy()
        centroid = pts.mean(axis=0)
        
        rr.log(
            "world/camera",
            rr.Pinhole(
                focal_length=500,
                width=1280,
                height=720,
            )
        )
        rr.log(
            "world/camera",
            rr.Transform3D(
                translation=[centroid[0] - 15, centroid[1] - 15, centroid[2] + 10]
            )
        )

        # Log point cloud - THIS IS WHAT WAS MISSING
        visualize_pointcloud(
            data.point_cloud.points,
            color_by=color_mode,
            entity_path=f"{entity_path}/point_cloud"
        )

        # Log bounding boxes
        visualize_boxes_3d(
            data.bboxes_3d,
            entity_path=f"{entity_path}/bboxes"
        )

        rr.log("frame_id", rr.TextLog(f"Frame: {data.frame_id}"))

    def log_sequence(
        self,
        sequence: Sequence,
        timeline: str = "frame"
    ) -> None:
        for i, frame in enumerate(sequence.frames):
            rr.set_time_sequence(timeline, i)
            sample = Sample(data=frame, split="val", dataset_name="kitti")
            self.log_sample(sample, entity_path=f"/sequence/frame_{i}")