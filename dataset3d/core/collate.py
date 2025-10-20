import torch
from dataclasses import dataclass
from typing import List , Optional , Dict
from dataset3d.core.unified_format import Sample, CalibrationData
@dataclass
class BatchStructure:
    """Explicit batch structure for training/inference
    
    All tensors use torch.float32 for points/boxes, torch.int64 for indices/labels.
    Variable-length data (points, boxes) stored as lists, with index tensors for reconstruction.
    """
    # Point cloud data
    points: List[torch.Tensor]  # List[N_i, C] where C >= 3 (xyz + features)
    num_points: torch.Tensor  # (B,) number of points per sample
    
    # Bounding box data
    bboxes: List[torch.Tensor]  # List[(M_i, 7+)] [x,y,z,l,w,h,yaw, ...] per sample
    bbox_labels: List[torch.Tensor]  # List[(M_i,)] class IDs per sample
    num_bboxes: torch.Tensor  # (B,) number of boxes per sample
    
    # Metadata
    frame_ids: List[str]  # Frame identifiers
    has_labels: torch.Tensor  # (B,) bool, whether ground truth available
    
    # Optional: Images (if loaded)
    images: Optional[Dict[str, List[torch.Tensor]]] = None  # {camera_name: List[(H,W,3)]}
    
    # Optional: Calibration
    calibrations: Optional[List[CalibrationData]] = None
    
    # Optional: Ego poses (for cooperative scenarios)
    ego_poses: Optional[List[torch.Tensor]] = None  # List[(4,4)] per sample
    
    # Optional: Sequence information
    sequence_ids: Optional[List[str]] = None
    timestamps: Optional[List[torch.Tensor]] = None  # List[(T,)] per sequence
    
    # Optional: Cooperative information
    agent_ids: Optional[List[List[str]]] = None  # List of agent ID lists per sample
    num_agents: Optional[torch.Tensor] = None  # (B,) number of agents per sample

def collate_3d_samples(batch: List[Sample]) -> BatchStructure:
    """Collate variable-size point clouds and boxes into explicit batch structure
    
    Handles:
    - Frame: Single frame, single agent
    - CooperativeFrame: Single frame, multiple agents
    - Sequence: Multiple frames, single agent
    - CooperativeSequence: Multiple frames, multiple agents
    
    Variable-Length Batching Strategy:
    - Point clouds: Keep as list, create batch_indices for pooling operations
    - Bounding boxes: Keep as list (each sample may have different numbers)
    - Sequences: Pad to max sequence length in batch if needed
    - Cooperative: Keep agents as nested structures
    
    Memory Optimization:
    - Images lazy-loaded, only included if already loaded
    - Calibration data shared by reference (not copied)
    
    Args:
        batch: List of Sample objects
        
    Returns:
        BatchStructure with organized tensors and metadata
    
    Future: Support segmentation masks collation
    """
    pass

def collate_sequences_padded(
    batch: List[Sample],
    max_length: Optional[int] = None,
    padding_mode: str = "replicate"  # "replicate", "zero", "none"
) -> BatchStructure:
    """Collate sequences with padding to uniform length
    
    Args:
        batch: List of Sample objects containing Sequence or CooperativeSequence
        max_length: Maximum sequence length (None = use longest in batch)
        padding_mode: How to pad shorter sequences
            - "replicate": Repeat last frame
            - "zero": Pad with empty point clouds and no boxes
            - "none": Raise error if lengths differ
    
    Returns:
        BatchStructure with padded sequences
    """
    pass

def collate_cooperative_aligned(
    batch: List[Sample],
    alignment_mode: str = "global"  # "global", "ego", "first_agent"
) -> BatchStructure:
    """Collate cooperative frames with coordinate alignment
    
    Args:
        batch: List of Sample objects containing CooperativeFrame
        alignment_mode: Coordinate frame for alignment
            - "global": Transform all to global/world frame
            - "ego": Transform all to first agent's frame
            - "first_agent": Use first agent in each sample as reference
    
    Returns:
        BatchStructure with aligned multi-agent data
    """
    pass