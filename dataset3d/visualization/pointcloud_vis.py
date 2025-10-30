import rerun as rr
import torch
import numpy as np
from typing import Optional , List
from dataset3d.core.unified_format import BBox3D



def visualize_pointcloud(
    points: torch.Tensor,  # (N, 3+)
    colors: Optional[torch.Tensor] = None,  # (N, 3) RGB or None
    color_by: str = "intensity",  # "intensity", "height", "z"
    entity_path: str = "/pointcloud"
) -> None:
    """Log point cloud to Rerun with optional coloring
    Future: Support segmentation visualization with per-point colors"""
    pts = points[:, :3].cpu().numpy()
    num_points = pts.shape[0]

    if colors is None:
        if color_by == "intensity" and points.shape[1] > 3:
            intensity = points[:, 3].cpu().numpy()
            colors = np.stack([intensity * 255] * 3, axis=-1).clip(0, 255)
        elif color_by in ["height", "z"]:
            z = pts[:, 2]
            norm_z = (z - z.min()) / (z.max() - z.min() + 1e-6)
            colors = np.stack([norm_z * 255, (1 - norm_z) * 255, np.zeros_like(norm_z)], axis=-1)
        else:
            colors = np.ones((num_points, 3)) * 255
    else:
        colors = colors.cpu().numpy()

    rr.log(entity_path, rr.Points3D(pts, colors=colors.astype(np.uint8), radii=0.02))


def visualize_boxes_3d(
    boxes: List[BBox3D],
    entity_path: str = "/boxes_3d"
) -> None:
    """Log 3D bounding boxes as line sets to Rerun"""
    print(f"🔍 Visualizing {len(boxes)} boxes")
    
    for i, bbox in enumerate(boxes):
        if bbox is None:
            print(f"❌ Box {i} is None, skipping")
            continue

        center = bbox.center.cpu().numpy() if hasattr(bbox.center, 'cpu') else bbox.center
        dims = bbox.dimensions.cpu().numpy() if hasattr(bbox.dimensions, 'cpu') else bbox.dimensions
        yaw = bbox.rotation_yaw
        
        print(f"✅ Box {i}: {bbox.class_name} at {center}")

        # Log each box with unique path
        box_entity = f"{entity_path}/box_{i}"
        rr.log(
            box_entity,
            rr.Boxes3D(
                centers=[center],
                half_sizes=[dims],
                rotation_axis_angles=[rr.RotationAxisAngle(axis=[0, 0, 1], angle=yaw)],
                colors=[255, 0, 0],
                labels=[bbox.class_name]
            )
        )
        print(f"📦 Logged to: {box_entity}")
    
    print(f"🎯 Finished logging {len(boxes)} boxes")