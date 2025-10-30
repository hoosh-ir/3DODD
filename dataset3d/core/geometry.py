import torch
from dataset3d.core.unified_format import BBox3D, Frame, CalibrationData
from typing import Optional


def transform_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Apply 4x4 transformation matrix to points (N,3) -> (N,3)"""
    assert points.ndim == 2 and points.shape[1] == 3, "points must be (N,3)"
    assert transform.shape == (4, 4), "transform must be (4,4)"

    ones = torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)
    points_h = torch.cat([points, ones], dim=1)  # (N,4)
    transformed = (transform @ points_h.T).T     # (N,4)
    transformed = transformed[:, :3] / transformed[:, 3:].clamp(min=1e-12)
    return transformed


def transform_bbox(
    bbox: BBox3D,
    transform: torch.Tensor,
    target_frame: str
) -> BBox3D:
    """Transform bbox to target coordinate frame"""
    center_homo = torch.cat([bbox.center, torch.tensor([1.0], device=bbox.center.device)])
    transformed_center_homo = transform @ center_homo
    transformed_center = transformed_center_homo[:3]

    transformed_bbox = BBox3D(
        center=transformed_center,
        dimensions=bbox.dimensions.clone(),
        rotation_yaw=bbox.rotation_yaw,
        coordinate_frame=target_frame,
        class_name=bbox.class_name,
        class_id=bbox.class_id,
        rotation_quat=bbox.rotation_quat.clone() if bbox.rotation_quat is not None else None,
        confidence=bbox.confidence,
        tracking_id=bbox.tracking_id,
        velocity=bbox.velocity.clone() if bbox.velocity is not None else None,
        attributes=bbox.attributes.copy()
    )
    return transformed_bbox


def transform_frame_to_frame(
    frame: Frame,
    target_frame: str,
    transform: Optional[torch.Tensor] = None,
    calib: Optional[CalibrationData] = None
) -> Frame:
    """Transform entire frame to target coordinate system"""
    if transform is None:
        if calib is None:
            raise ValueError("Either transform or calib must be provided.")
        transform = get_transform_between_frames(
            frame.coordinate_frame, target_frame, calib
        )

    transformed_points = transform_points(frame.points, transform)
    transformed_boxes = [
        transform_bbox(b, transform, target_frame) for b in frame.boxes_3d
    ]

    transformed_frame = Frame(
        data=frame.data,
        points=transformed_points,
        boxes_3d=transformed_boxes,
        coordinate_frame=target_frame
    )
    return transformed_frame


def lidar_to_camera(points: torch.Tensor, calib: CalibrationData, camera_name: str) -> torch.Tensor:
    """Transform points from LiDAR to camera coordinate frame"""
    T_lidar_to_cam = torch.as_tensor(getattr(calib, "Tr_velo_to_cam"), dtype=points.dtype, device=points.device)
    if T_lidar_to_cam.shape == (3, 4):
        T_lidar_to_cam = torch.cat([T_lidar_to_cam, torch.tensor([[0, 0, 0, 1]], dtype=points.dtype, device=points.device)], dim=0)
    return transform_points(points, T_lidar_to_cam)


def camera_to_image(points: torch.Tensor, intrinsic: torch.Tensor) -> torch.Tensor:
    """Project 3D camera points to 2D image plane (N,3) -> (N,2)"""
    assert intrinsic.shape == (3, 3), "Intrinsic must be (3,3)"
    points_h = points.T  # (3,N)
    proj = intrinsic @ points_h
    u = proj[0, :] / proj[2, :].clamp(min=1e-12)
    v = proj[1, :] / proj[2, :].clamp(min=1e-12)
    return torch.stack([u, v], dim=1)


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (4,) [w,x,y,z] to rotation matrix (3,3)"""
    assert q.shape[-1] == 4, "Quaternion must have shape (4,)"
    w, x, y, z = q
    R = torch.tensor([
        [1 - 2 * (y**2 + z**2), 2 * (x*y - z*w),     2 * (x*z + y*w)],
        [2 * (x*y + z*w),       1 - 2 * (x**2 + z**2), 2 * (y*z - x*w)],
        [2 * (x*z - y*w),       2 * (y*z + x*w),     1 - 2 * (x**2 + y**2)]
    ], dtype=q.dtype, device=q.device)
    return R


def rotation_matrix_to_euler(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix (3,3) to euler angles (3,) [roll,pitch,yaw]"""
    assert R.shape == (3, 3)
    sy = torch.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        roll = torch.atan2(R[2, 1], R[2, 2])
        pitch = torch.atan2(-R[2, 0], sy)
        yaw = torch.atan2(R[1, 0], R[0, 0])
    else:
        roll = torch.atan2(-R[1, 2], R[1, 1])
        pitch = torch.atan2(-R[2, 0], sy)
        yaw = torch.tensor(0., dtype=R.dtype)
    return torch.stack([roll, pitch, yaw])


def get_transform_between_frames(
    from_frame: str,
    to_frame: str,
    calib: CalibrationData,
    ego_pose: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Get transformation matrix between two coordinate frames"""
    def as_44(M):
        M = torch.as_tensor(M, dtype=torch.float32)
        if M.shape == (3, 4):
            M = torch.cat([M, torch.tensor([[0, 0, 0, 1]], dtype=M.dtype)], dim=0)
        return M

    if from_frame == to_frame:
        return torch.eye(4)

    if hasattr(calib, "Tr_velo_to_cam"):
        T_lidar_to_cam = as_44(calib.Tr_velo_to_cam)
        if from_frame.lower().startswith("lidar") and to_frame.lower().startswith("cam"):
            return T_lidar_to_cam
        elif from_frame.lower().startswith("cam") and to_frame.lower().startswith("lidar"):
            return torch.linalg.inv(T_lidar_to_cam)

    if ego_pose is not None:
        if from_frame == "ego" and to_frame == "world":
            return ego_pose
        if from_frame == "world" and to_frame == "ego":
            return torch.linalg.inv(ego_pose)

    raise ValueError(f"No transform available from {from_frame} to {to_frame}.")
