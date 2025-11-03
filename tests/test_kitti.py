import os
import sys
import torch
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset3d.datasets.kitti_dataset import KITTIDataset
from dataset3d.core.unified_format import Frame, Sample

DATA_ROOT = "/media/sina/New Volume1/data/kitti"


@pytest.fixture(scope="module")
def dataset():
    """Load KITTI dataset once for all tests"""
    return KITTIDataset(
        data_root=DATA_ROOT,
        split="train",
        target_frame="lidar",
        modalities=["lidar"]
    )


def test_sample_loading(dataset):
    """Ensure sample list is loaded correctly"""
    assert len(dataset.sample_list) > 0, "Dataset sample list is empty"


def test_load_first_sample(dataset):
    """Test loading of first sample"""
    sample = dataset[0]
    assert isinstance(sample, Sample)
    assert isinstance(sample.data, Frame)
    assert sample.data.point_cloud.points.shape[1] >= 3, "Point cloud must have at least 3 coordinates"
    assert isinstance(sample.data.has_labels, bool)


def test_bboxes_have_valid_format(dataset):
    """Test bbox properties and class mapping"""
    sample = dataset[0]

    if not sample.data.has_labels or len(sample.data.bboxes_3d) == 0:
        pytest.skip("No bounding boxes available in this frame.")

    for bbox in sample.data.bboxes_3d:
        if bbox is None:
            pytest.skip("Encountered None-type bbox in dataset.")
        assert isinstance(bbox.center, torch.Tensor), "bbox.center should be a Tensor"
        assert bbox.center.shape == (3,), "bbox.center must have shape (3,)"
        assert isinstance(bbox.class_name, str), "bbox.class_name should be a string"
        assert isinstance(bbox.class_id, int), "bbox.class_id should be an integer"


def test_coordinate_frame_matches(dataset):
    """Test that the target_frame is respected"""
    sample = dataset[0]
    frame = sample.data
    assert frame.point_cloud.coordinate_frame == dataset.target_frame, \
        f"Expected frame {dataset.target_frame}, got {frame.point_cloud.coordinate_frame}"


def test_calibration_integrity(dataset):
    """Calibration matrices should exist and have correct shape"""
    sample = dataset[0]
    calib = sample.data.calibration
    assert calib.lidar_to_camera is not None, "Calibration must include lidar_to_camera"
    for k, mat in calib.lidar_to_camera.items():
        assert isinstance(mat, torch.Tensor), f"Calibration matrix for {k} must be a torch.Tensor"
        assert mat.shape == (4, 4), f"Calibration matrix for {k} must be 4x4"
