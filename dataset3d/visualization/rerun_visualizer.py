import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import rerun as rr
import torch
import numpy as np
from dataset3d.core.unified_format import Sample, Sequence, Frame
from dataset3d.visualization.pointcloud_vis import visualize_pointcloud, visualize_boxes_3d


class RerunVisualizer:
    """Main visualizer class wrapping Rerun SDK"""
    
    def __init__(self, recording_name: str = "3dodd"):
        """Initialize Rerun recording"""
        rr.init(recording_name, spawn=True)

    def log_sample(
        self,
        sample: Sample,
        entity_path: str = "/",
        color_mode: str = "intensity"  # "intensity", "height", "class"
    ) -> None:
        """Log Sample to Rerun with automatic entity structure"""
        data = sample.data
        if not isinstance(data, Frame):
            raise TypeError("RerunVisualizer currently supports only single-agent Frame visualization.")

        visualize_pointcloud(
            data.point_cloud.points,
            color_by=color_mode,
            entity_path=f"{entity_path}/point_cloud"
        )

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
        """Log temporal sequence with timeline"""
        for i, frame in enumerate(sequence.frames):
            rr.set_time_sequence(timeline, i)
            sample = Sample(data=frame, split="val", dataset_name="kitti")
            self.log_sample(sample, entity_path=f"/sequence/frame_{i}")
