import sys
import os
sys.path.append(os.path.abspath('.'))
import numpy as np
import rerun as rr
import torch
from typing import Tuple
from dataset3d.core.unified_format import Frame


def visualize_bev(
    frame: Frame,
    range_x: Tuple[float, float] = (-50, 50),
    range_y: Tuple[float, float] = (-50, 50),
    resolution: float = 0.1,
    entity_path: str = "/bev"
) -> None:
    """Create bird's-eye view 2D projection of points and boxes"""
    points = frame.point_cloud.points.cpu().numpy()
    x, y = points[:, 0], points[:, 1]

    mask = (
        (x > range_x[0]) & (x < range_x[1]) &
        (y > range_y[0]) & (y < range_y[1])
    )
    x, y = x[mask], y[mask]

    bev_w = int((range_x[1] - range_x[0]) / resolution)
    bev_h = int((range_y[1] - range_y[0]) / resolution)

    bev_img = np.zeros((bev_h, bev_w, 3), dtype=np.uint8)

    # Normalize coordinates to BEV image space
    px = ((x - range_x[0]) / (range_x[1] - range_x[0]) * bev_w).astype(int)
    py = ((y - range_y[0]) / (range_y[1] - range_y[0]) * bev_h).astype(int)
    bev_img[py, px] = [255, 255, 255]

    # Overlay 2D boxes
    for bbox in frame.bboxes_3d:
        if bbox is None:
            continue
        cx, cy = bbox.center[:2].cpu().numpy()
        l, w = bbox.dimensions[:2].cpu().numpy()
        yaw = bbox.rotation_yaw
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)

        corners = np.array([
            [l/2, w/2],
            [l/2, -w/2],
            [-l/2, -w/2],
            [-l/2, w/2]
        ])
        R = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
        rotated = corners @ R.T + np.array([cx, cy])

        px = ((rotated[:, 0] - range_x[0]) / (range_x[1] - range_x[0]) * bev_w).astype(int)
        py = ((rotated[:, 1] - range_y[0]) / (range_y[1] - range_y[0]) * bev_h).astype(int)
        for i in range(4):
            rr.log(
                f"{entity_path}/boxes/{bbox.class_name}/{i}",
                rr.LineStrips2D(
                    [[(px[i], py[i]), (px[(i+1)%4], py[(i+1)%4])]],
                    colors=[[255, 0, 0]]
                )
            )

    # This is the key: use 2D viewer (Grid) instead of 3D scene
    rr.log(f"{entity_path}/bev_image", rr.Image(bev_img))
