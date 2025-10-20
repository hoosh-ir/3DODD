import torch
from core.unified_format import BBox3D

def bbox_iou_3d(boxes1: torch.Tensor, boxes2: torch.Tensor, 
                mode: str = 'iou') -> torch.Tensor:
    """Compute 3D IoU between boxes (M,7) x (N,7) -> (M,N). Format: [x,y,z,l,w,h,yaw]"""

def nms_3d(boxes: torch.Tensor, scores: torch.Tensor, 
           threshold: float = 0.5) -> torch.Tensor:
    """3D Non-Maximum Suppression, returns indices of kept boxes"""

def corners_from_bbox(bbox: BBox3D) -> torch.Tensor:
    """Convert bbox parameters to 8 corner points (8,3)"""

def rotate_points_along_z(points: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rotate points around Z-axis for yaw rotation"""