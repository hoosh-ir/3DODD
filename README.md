# 3D Object Detection Dataset Library (3DODD)

## Overview

This library provides a unified interface for 3D object detection datasets with support for:
- **Multiple Datasets**: KITTI, nuScenes, Waymo, ONCE, DAIR-V2X, OPV2V
- **Flexible Coordinate Systems**: Each dataset uses its native coordinate frame; transformations provided
- **Cooperative Detection**: Multi-agent scenarios with coordinate transformations
- **Temporal Sequences**: Tracking and temporal modeling support
- **Framework Interoperability**: Converters for MMDetection3D, OpenPCDet
- **Training Ready**: Explicit batch structures, transforms, memory optimization

**Design Status**: ✅ All critical issues resolved (see "Critical Design Fixes Applied" section at end)

## Project Structure

Create a modular library with the following structure:

```
   /
├── dataset3d/               # Main package directory
│   ├── __init__.py
│   ├── core/                # Core abstractions and unified format
│   ├── datasets/            # Dataset loaders (KITTI, nuScenes, Waymo, ONCE, DAIR-V2X, OPV2V)
│   ├── converters/          # Bidirectional format converters (datasets + frameworks)
│   ├── visualization/       # Rerun-based visualization toolkit
│   ├── cooperative/         # Coordinate transformations for cooperative detection
│   └── utils/              # Geometry, I/O, bounding box utilities
├── tests/                   # Unit tests (outside main package)
├── examples/                # Usage examples and tutorials (outside main package)
├── docs/                    # Documentation
├── setup.py
├── requirements.txt
└── README.md
```

## Implementation Phases

### Phase 1: Core Infrastructure

**Purpose**: Establish the foundation of the library with a unified intermediate format that serves as a translation layer between different dataset formats and frameworks.

**1.1 Unified Format Definition** (`core/unified_format.py`)


**Coordinate Frame Conventions** (`core/coordinate_frames.py`):

```python
class CoordinateFrame:
    """Coordinate frame definitions and conventions
    
    Supported coordinate frames with their conventions:
    
    1. "lidar": LiDAR sensor frame
       - Origin: LiDAR sensor position
       - Common conventions:
         * X: Forward, Y: Left, Z: Up (most common: Waymo, nuScenes)
         * X: Forward, Y: Right, Z: Down (some systems)
       - Use dataset metadata to specify exact convention
    
    2. "camera": Camera sensor frame (rectified)
       - Origin: Camera optical center
       - Common conventions:
         * X: Right, Y: Down, Z: Forward (KITTI, OpenCV standard)
         * X: Right, Y: Up, Z: Forward (some systems)
       - Use dataset metadata to specify exact convention
    
    3. "ego": Ego vehicle frame
       - Origin: Vehicle center (typically center of rear axle or vehicle center)
       - X: Forward, Y: Left, Z: Up (right-handed, most common)
    
    4. "global": Global/world frame
       - Origin: Map/world origin
       - Typically: X: East, Y: North, Z: Up (ENU convention)
       - Or: X: North, Y: East, Z: Down (NED convention)
    
    Each dataset loader should document its coordinate frame conventions.
    """
    
    VALID_FRAMES = ["lidar", "camera", "ego", "global"]
    
    # Dataset-specific coordinate conventions (axis directions)
    DATASET_CONVENTIONS = {
        "kitti": {
            "camera": {"x": "right", "y": "down", "z": "forward"},  # Rectified camera
            "lidar": {"x": "forward", "y": "left", "z": "up"}       # Velodyne
        },
        "nuscenes": {
            "lidar": {"x": "forward", "y": "left", "z": "up"},
            "camera": {"x": "right", "y": "down", "z": "forward"},
            "ego": {"x": "forward", "y": "left", "z": "up"},
            "global": {"x": "east", "y": "north", "z": "up"}  # ENU
        },
        "waymo": {
            "lidar": {"x": "forward", "y": "left", "z": "up"},
            "camera": {"x": "right", "y": "down", "z": "forward"},
            "ego": {"x": "forward", "y": "left", "z": "up"}
        },
        # Add other datasets...
    }
    
    @staticmethod
    def validate_frame(frame: str) -> bool:
        """Validate coordinate frame name"""
        return frame in CoordinateFrame.VALID_FRAMES
    
    @staticmethod
    def get_convention(dataset_name: str, frame: str) -> Dict[str, str]:
        """Get axis convention for dataset and frame
        
        Returns:
            Dict with 'x', 'y', 'z' keys mapping to direction strings
        """
        pass
```

**JSON Serialization Support**:
- Custom encoder/decoder for torch.Tensor ↔ list conversion
- Validation schema using pydantic or dataclass validators

**Class Registry** (`core/class_registry.py`):

```python
class ClassRegistry:
    """Global class taxonomy for multi-dataset training
    
    Maps dataset-specific class names/IDs to unified class IDs.
    Enables joint training across datasets with different class definitions.
    """
    
    # Unified class taxonomy (extensible)
    UNIFIED_CLASSES = [
        "Car", "Truck", "Bus", "Trailer", "Construction_Vehicle",
        "Pedestrian", "Cyclist", "Motorcyclist",
        "Traffic_Cone", "Barrier",
        "Background"  # For negative samples
    ]
    
    # Dataset-specific mappings
    DATASET_MAPPINGS = {
        "kitti": {
            "Car": "Car",
            "Van": "Car",
            "Truck": "Truck",
            "Pedestrian": "Pedestrian",
            "Person_sitting": "Pedestrian",
            "Cyclist": "Cyclist",
            "Tram": "Bus",
            "Misc": "Background",
            "DontCare": "Background"
        },
        "nuscenes": {
            "car": "Car",
            "truck": "Truck",
            "bus": "Bus",
            "trailer": "Trailer",
            "construction_vehicle": "Construction_Vehicle",
            "pedestrian": "Pedestrian",
            "bicycle": "Cyclist",
            "motorcycle": "Motorcyclist",
            "traffic_cone": "Traffic_Cone",
            "barrier": "Barrier"
        },
        # Add mappings for other datasets: waymo, once, dair_v2x, opv2v
    }
    
    def __init__(self):
        self.class_to_id = {name: idx for idx, name in enumerate(self.UNIFIED_CLASSES)}
        self.id_to_class = {idx: name for name, idx in self.class_to_id.items()}
    
    def map_dataset_class_to_unified(
        self, 
        dataset_name: str, 
        dataset_class: str
    ) -> Tuple[str, int]:
        """Map dataset-specific class name to unified class name and ID
        
        Returns:
            (unified_class_name, unified_class_id)
        """
        pass
    
    def get_unified_id(self, unified_class_name: str) -> int:
        """Get unified class ID from unified class name"""
        pass
    
    def filter_classes(
        self, 
        bboxes: List[BBox3D], 
        allowed_classes: List[str]
    ) -> List[BBox3D]:
        """Filter bboxes to only include specified unified classes"""
        pass
```

**1.2 Base Dataset Class** (`core/base_dataset.py`)

**Class Structure**:

```python
class Base3DDataset(torch.utils.data.Dataset):
    """Base dataset class for 3D object detection datasets
    
    Memory Management:
    - Images are lazy-loaded by default (only loaded when accessed)
    - Point clouds are always loaded (required for most tasks)
    - Use pickle cache for faster loading after first run
    
    Pickle Cache Versioning:
    - Cache includes dataset version hash based on: data_root, split, modalities
    - Automatically invalidates if source data timestamp > cache timestamp
    - Version stored in pickle metadata
    
    DataLoader Recommendations:
    - num_workers=4-8 for non-pickled data
    - num_workers=0-2 for pickled data (already fast)
    - pin_memory=True for GPU training
    - persistent_workers=True for sequences
    """
    
    DATASET_VERSION = "1.0.0"  # Increment when format changes
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        transform: Optional[Callable[[Sample], Sample]] = None,
        load_images: bool = False,  # Load images immediately (memory intensive)
        lazy_load_images: bool = True,  # Lazy load images on access (recommended)
        sequence_length: Optional[int] = None,  # For temporal sampling
        modalities: List[str] = ["lidar"],  # ["lidar", "camera", "radar"]
        target_frame: Optional[str] = None,  # Target coordinate frame ("lidar", "camera", "ego", "global")
        use_pickle: bool = False,  # Use preprocessed pickle files
        pickle_path: Optional[str] = None,  # Custom pickle file path
        auto_create_cache: bool = False,  # Automatically create pickle cache on first run
        cache_version_check: bool = True,  # Validate cache version and timestamps
        class_registry: Optional[ClassRegistry] = None,  # For unified class mapping
    ):
        """Initialize dataset with configurable sensor modalities and optional pickle caching
        
        Args:
            data_root: Root directory of dataset
            split: "train", "val", or "test"
            transform: Optional transform function (see utils/transforms.py)
            load_images: Load all images into memory (not recommended for large datasets)
            lazy_load_images: Load images on-demand per sample (recommended)
            sequence_length: Number of frames per sequence (None = single frames)
            modalities: Sensor modalities to load
            target_frame: Target coordinate frame for output data (None = use dataset native frame)
                Options: "lidar", "camera", "ego", "global"
                If specified, all data will be transformed to this frame
            use_pickle: Load from preprocessed pickle cache
            pickle_path: Custom pickle cache location (default: {data_root}/cache/)
            auto_create_cache: Create cache automatically if missing
            cache_version_check: Validate cache is up-to-date
            class_registry: ClassRegistry for unified class mapping
        """
    
    @abstractmethod
    def _load_sample_list(self) -> List[str]:
        """Load list of sample identifiers from dataset metadata"""
        pass
    
    @abstractmethod
    def load_raw_data(self, idx: int) -> Dict[str, Any]:
        """Load raw data from disk in native dataset format"""
        pass
    
    @abstractmethod
    def to_unified(self, raw_data: Dict[str, Any]) -> Sample:
        """Convert native format to unified Sample format"""
        pass
    
    def __len__(self) -> int:
        return len(self.sample_list)
    
    def __getitem__(self, idx: int) -> Sample:
        """Get sample with caching, loading, conversion, and transform
        
        Loading order:
        1. Check pickle cache if enabled
        2. Load raw data from disk
        3. Convert to unified format
        4. Apply transforms if specified
        5. Lazy-load images if requested
        
        Returns:
            Sample with requested modalities and transforms applied
        """
        pass
        
    def create_pickle_cache(
        self, 
        output_path: Optional[str] = None,
        num_workers: int = 4,
        show_progress: bool = True
    ) -> None:
        """Create pickle cache of all samples for faster loading
        
        Cache includes:
        - Preprocessed samples in unified format
        - Dataset version and configuration hash
        - Creation timestamp
        - Source data checksums (optional, for validation)
        
        Args:
            output_path: Where to save cache (default: {data_root}/cache/{split}.pkl)
            num_workers: Parallel workers for cache creation
            show_progress: Show progress bar during caching
        """
        pass
    
    def load_pickle_cache(self, pickle_path: Optional[str] = None) -> None:
        """Load preprocessed samples from pickle file
        
        Validates:
        - Cache version matches DATASET_VERSION
        - Configuration hash matches current settings
        - Source data not modified after cache creation (if cache_version_check=True)
        
        Raises:
            ValueError: If cache is invalid or outdated
        """
        pass
    
    def _validate_cache(self, cache_metadata: Dict[str, Any]) -> bool:
        """Validate pickle cache is compatible and up-to-date
        
        Returns:
            True if cache is valid, False otherwise
        """
        pass
    
    def _compute_config_hash(self) -> str:
        """Compute hash of dataset configuration for cache validation"""
        pass
    
    @staticmethod
    def get_dataloader_config(
        batch_size: int = 1,
        num_workers: int = 4,
        use_pickle: bool = False
    ) -> Dict[str, Any]:
        """Get recommended DataLoader configuration
        
        Returns:
            Dict with DataLoader kwargs optimized for this dataset
        """
        return {
            "batch_size": batch_size,
            "num_workers": 2 if use_pickle else num_workers,
            "pin_memory": True,
            "collate_fn": collate_3d_samples,
            "persistent_workers": num_workers > 0,
            "prefetch_factor": 2 if num_workers > 0 else None,
        }

```

**Collate Function** (`core/collate.py`):

```python
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
```

**1.3 Utility Modules**

**`utils/geometry.py`** - 3D transformations and coordinate systems:

```python
def transform_points(points: torch.Tensor, transform: torch.Tensor) -> torch.Tensor:
    """Apply 4x4 transformation matrix to points (N,3) -> (N,3)
    
    Args:
        points: (N, 3) points in source frame
        transform: (4, 4) SE(3) transformation matrix
        
    Returns:
        (N, 3) points in target frame
    """
    pass

def transform_bbox(
    bbox: BBox3D, 
    transform: torch.Tensor,
    target_frame: str
) -> BBox3D:
    """Transform bbox to target coordinate frame
    
    Args:
        bbox: BBox3D in source frame
        transform: (4, 4) transformation from source to target
        target_frame: Name of target coordinate frame
        
    Returns:
        BBox3D in target frame with updated coordinate_frame
    """
    pass

def transform_frame_to_frame(
    frame: Frame,
    target_frame: str,
    transform: Optional[torch.Tensor] = None,
    calib: Optional[CalibrationData] = None
) -> Frame:
    """Transform entire frame to target coordinate system
    
    Args:
        frame: Input frame
        target_frame: Target coordinate frame name
        transform: (4, 4) transformation matrix (if known)
        calib: CalibrationData for computing transform (if transform not provided)
        
    Returns:
        Frame with all data (points, boxes) transformed to target_frame
    """
    pass

def lidar_to_camera(points: torch.Tensor, calib: CalibrationData, camera_name: str) -> torch.Tensor:
    """Transform points from LiDAR to camera coordinate frame
    
    Args:
        points: (N, 3) points in LiDAR frame
        calib: CalibrationData with lidar_to_camera transforms
        camera_name: Which camera to transform to
        
    Returns:
        (N, 3) points in camera frame
    """
    pass

def camera_to_image(points: torch.Tensor, intrinsic: torch.Tensor) -> torch.Tensor:
    """Project 3D camera points to 2D image plane (N,3) -> (N,2)
    
    Args:
        points: (N, 3) points in camera frame
        intrinsic: (3, 3) camera intrinsic matrix
        
    Returns:
        (N, 2) pixel coordinates
    """
    pass

def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (4,) [w,x,y,z] to rotation matrix (3,3)"""
    pass

def rotation_matrix_to_euler(R: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix (3,3) to euler angles (3,) [roll,pitch,yaw]"""
    pass

def get_transform_between_frames(
    from_frame: str,
    to_frame: str,
    calib: CalibrationData,
    ego_pose: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Get transformation matrix between two coordinate frames
    
    Args:
        from_frame: Source coordinate frame name
        to_frame: Target coordinate frame name
        calib: CalibrationData with transformation matrices
        ego_pose: (4, 4) ego pose in global frame (for global transforms)
        
    Returns:
        (4, 4) transformation matrix from source to target
    """
    pass
```

**`utils/bbox.py`** - 3D bounding box operations:

```python
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
```

**`utils/io.py`** - File I/O:

```python
def load_bin_pointcloud(file_path: str) -> np.ndarray:
    """Load KITTI-style .bin point cloud file"""

def load_pcd_pointcloud(file_path: str) -> np.ndarray:
    """Load .pcd format point cloud file"""

def save_to_json(data: Sample, file_path: str) -> None:
    """Serialize Sample to JSON with tensor encoding"""

def load_from_json(file_path: str) -> Sample:
    """Deserialize Sample from JSON"""
```

**`utils/transforms.py`** - Transform interface:

```python
class Transform(ABC):
    """Abstract transform interface for augmentation pipelines
    
    Transform Contract:
    1. Must handle all Sample types: Frame, CooperativeFrame, Sequence, CooperativeSequence
    2. Must synchronize transformations across:
       - Point clouds and bounding boxes (spatial consistency)
       - Multiple cameras (if multi-view)
       - Multiple agents (if cooperative)
       - Multiple frames (if sequence)
    3. Must preserve data types and coordinate systems
    4. Should be deterministic if random_state is set
    5. Should have inverse() method for validation (optional)
    
    Random State Management:
    - Set random_state in __init__ for reproducibility
    - Use self.rng = np.random.RandomState(seed) for deterministic augmentation
    """
    
    def __init__(self, random_state: Optional[int] = None):
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state) if random_state is not None else np.random
    
    @abstractmethod
    def __call__(self, sample: Sample) -> Sample:
        """Apply transform to sample
        
        Args:
            sample: Input sample (any type)
            
        Returns:
            Transformed sample (same type as input)
        """
        pass
    
    def inverse(self, sample: Sample) -> Sample:
        """Inverse transform (optional, for validation)
        
        Not all transforms are invertible (e.g., random drop, noise)
        """
        raise NotImplementedError(f"{self.__class__.__name__} is not invertible")

class Compose(Transform):
    """Compose multiple transforms sequentially"""
    def __init__(self, transforms: List[Transform]):
        ...
    def __call__(self, sample: Sample) -> Sample:
        """Normalize point features: (x - mean) / std"""
        pass
```

---

### Phase 2: Dataset Loaders

**Purpose**: Provide unified access to diverse 3D detection datasets while respecting each dataset's unique characteristics and conventions.

**2.1 Standard Autonomous Driving Datasets**

**KITTI** (`datasets/kitti.py`):

**nuScenes** (`datasets/nuscenes.py`):

```python
class NuScenesDataset(Base3DDataset):
    """nuScenes Dataset with tracking and sequences"""
    
    def __init__(
        self,
        ...,
        modalities: List[str] = ["lidar"],  # ["lidar", "camera", "radar"]
        use_sequences: bool = False,
        sequence_length: int = 5
    ):
        """Initialize with nuScenes SDK, configurable modalities and temporal sequences"""
    
    def load_raw_data(self, idx: int) -> Dict[str, Any]:
        """Load sample data, annotations, ego pose, calibration; optionally consecutive frames"""
    
    def to_unified(self, raw_data: Dict[str, Any]) -> Sample:
        """Convert to Frame or Sequence with tracking IDs and velocity"""
```

**Waymo** (`datasets/waymo.py`):

```python
class WaymoDataset(Base3DDataset):
    """Waymo Open Dataset"""
    
    def __init__(
        self,
        ...,
        modalities: List[str] = ["lidar"],  # ["lidar", "camera", "radar"]
        use_sequences: bool = False,
        sequence_length: int = 5
    ):
        """Initialize Waymo dataset with configurable modalities"""
    
    def _load_sample_list(self) -> List[str]:
        """Parse TFRecord files and extract frame contexts"""
    
    def load_raw_data(self, idx: int) -> Dict[str, Any]:
        """Read TFRecord with laser labels, points, and difficulty"""
    
    def to_unified(self, raw_data: Dict[str, Any]) -> Sample:
        """Parse protobuf and convert Waymo coords to unified format"""

**ONCE** (`datasets/once.py`):

```python
class ONCADataset(Base3DDataset):
    """ONCE Dataset"""
    
    def __init__(
        self,
        ...,
        modalities: List[str] = ["lidar"],  # ["lidar", "camera", "radar"]
        use_sequences: bool = False,
        sequence_length: int = 5
    ):
        """Initialize ONCE dataset with configurable modalities"""
    
    def load_raw_data(self, idx: int) -> Dict[str, Any]:
        """Load from sequence structure: {seq_id}/{frame_id}"""
    
    def to_unified(self, raw_data: Dict[str, Any]) -> Sample:
        """Parse ONCE JSON annotations to unified format"""
```

**2.2 Cooperative Detection Datasets**

**DAIR-V2X** (`datasets/dair_v2x.py`):

```python
class DAIRV2XDataset(Base3DDataset):
    """DAIR-V2X Vehicle-to-Infrastructure dataset"""
    
    def __init__(
        self,
        ...,
        agents: List[str] = ["vehicle", "infrastructure"],  # Which agents to load
        modalities: List[str] = ["lidar"],  # Sensor modalities per agent
        use_sequences: bool = False,
        sequence_length: int = 5
    ):
        """Initialize DAIR-V2X with configurable agents and modalities"""
    
    def load_raw_data(self, idx: int) -> Dict[str, Any]:
        """Load vehicle and infrastructure data, optionally temporal"""
    
    def to_unified(self, raw_data: Dict[str, Any]) -> Sample:
        """Convert to CooperativeFrame or Sequence based on agents and modalities"""
```

**OPV2V** (`datasets/opv2v.py`):

```python
class OPV2VDataset(Base3DDataset):
    """OPV2V Vehicle-to-Vehicle dataset"""
    
    def __init__(
        self,
        ...,
        agents: List[str] = None,  # None = all vehicles, or specify subset
        modalities: List[str] = ["lidar"],  # Sensor modalities per agent
        use_sequences: bool = False,
        sequence_length: int = 5
    ):
        """Initialize OPV2V with configurable vehicle agents and modalities"""
    
    def load_raw_data(self, idx: int) -> Dict[str, Any]:
        """Load multi-agent V2V scenario data with specified agents and modalities"""
    
    def to_unified(self, raw_data: Dict[str, Any]) -> Sample:
        """Convert to CooperativeFrame with specified agent data"""
```

---

### Phase 3: Format Converters

**Purpose**: Enable seamless interoperability between datasets and frameworks without forcing users into a single format.

**3.1 Framework Export** (`converters/`)

**MMDetection3D** (`converters/to_mmdet3d.py`):

```python
def convert_to_mmdet3d(
    dataset: Base3DDataset, 
    output_path: str,
    workers: int = 4
) -> None:
    """Convert dataset to MMDetection3D pickle format with parallel workers"""

def sample_to_mmdet3d_dict(sample: Sample) -> Dict[str, Any]:
    """Convert single Sample to MMDet3D dict format"""
```

**OpenPCDet** (`converters/to_openpcdet.py`):

```python
def convert_to_openpcdet(
    dataset: Base3DDataset,
    output_path: str,
    cfg: Dict[str, Any]
) -> None:
    """Convert dataset to OpenPCDet format with infos_{split}.pkl"""
```

**3.2 Dataset Format Export** (`converters/`)

**KITTI Export** (`converters/to_kitti.py`):

```python
def convert_to_kitti(
    samples: List[Sample],
    output_dir: str,
    split: str = "training"
) -> None:
    """Export to KITTI directory structure (velodyne/, label_2/, calib/, ImageSets/)"""

def frame_to_kitti_label(frame: Frame) -> List[str]:
    """Convert Frame to KITTI label format strings"""
```

**nuScenes Export** (`converters/to_nuscenes.py`):

```python
def convert_to_nuscenes(
    sequences: List[Sequence],
    output_dir: str
) -> None:
    """Export to nuScenes JSON structure with relational tables"""
```

**3.3 Batch Processing** (`converters/batch_convert.py`):

```python
class BatchConverter:
    """Batch dataset conversion with validation"""
    
    def __init__(self, num_workers: int = 4):
        ...
    
    def convert_dataset(
        self,
        source_dataset: Base3DDataset,
        target_format: str,  # "mmdet3d", "openpcdet", "kitti", "nuscenes"
        output_path: str,
        validate: bool = True
    ) -> None:
        """Parallel conversion with progress bar and optional validation"""
    
    def validate_conversion(
        self,
        original: Sample,
        converted_back: Sample
    ) -> Dict[str, float]:
        """Validate conversion accuracy via round-trip metrics"""
```

---

### Phase 4: Visualization Toolkit

**Purpose**: Provide powerful, interactive 3D visualization using Rerun for debugging, exploration, and presentation.

**4.1 Core Visualization** (`visualization/`)

**Rerun Visualizer** (`visualization/rerun_visualizer.py`):

```python
class RerunVisualizer:
    """Main visualizer class wrapping Rerun SDK"""
    
    def __init__(self, recording_name: str = "3dodd"):
        """Initialize Rerun recording"""
    
    def log_sample(
        self,
        sample: Sample,
        entity_path: str = "/",
        color_mode: str = "intensity"  # "intensity", "height", "class"
    ) -> None:
        """Log Sample to Rerun with automatic entity structure"""
    
    def log_sequence(
        self,
        sequence: Sequence,
        timeline: str = "frame"
    ) -> None:
        """Log temporal sequence with timeline"""
```

**Point Cloud Visualization** (`visualization/pointcloud_vis.py`):

```python
def visualize_pointcloud(
    points: torch.Tensor,  # (N, 3+)
    colors: Optional[torch.Tensor] = None,  # (N, 3) RGB or None
    color_by: str = "intensity",  # "intensity", "height", "z"
    entity_path: str = "/pointcloud"
) -> None:
    """Log point cloud to Rerun with optional coloring
    Future: Support segmentation visualization with per-point colors"""

def visualize_boxes_3d(
    boxes: List[BBox3D],
    entity_path: str = "/boxes_3d"
) -> None:
    """Log 3D bounding boxes as line sets to Rerun"""
```

**BEV Visualization** (`visualization/bev_vis.py`):

```python
def visualize_bev(
    frame: Frame,
    range_x: Tuple[float, float] = (-50, 50),
    range_y: Tuple[float, float] = (-50, 50),
    resolution: float = 0.1,
    entity_path: str = "/bev"
) -> None:
    """Create bird's-eye view 2D projection of points and boxes"""
```

**Camera Projection** (`visualization/camera_projection.py`):

```python
def project_boxes_to_image(
    frame: Frame,
    camera_name: str,
    entity_path: str = "/camera"
) -> None:
    """Project 3D boxes onto camera image using calibration"""
```

**4.2 Cooperative Visualization** (`visualization/cooperative_vis.py`):

```python
def visualize_cooperative_frame(
    coop_frame: CooperativeFrame,
    show_transforms: bool = True,
    show_comm_range: bool = True
) -> None:
    """Visualize multi-agent frame with all agents in global coordinate frame"""

def visualize_cooperative_batch(
    batch: Dict[str, Any],  # From collate_3d_samples
    timeline: str = "cooperative"
) -> None:
    """Visualize batch of cooperative frames with timeline"""
```

---

### Phase 5: Cooperative Detection Support

**Purpose**: Provide essential utilities for multi-agent coordinate transformations without prescribing fusion algorithms.

**Coordinate Transformations** (`cooperative/transforms.py`):

```python
def agent_to_agent_transform(
    source_frame: Frame,
    target_frame: Frame
) -> torch.Tensor:
    """Compute (4,4) transformation matrix from source agent to target agent"""

def transform_frame_to_agent(
    frame: Frame,
    target_agent_pose: torch.Tensor
) -> Frame:
    """Transform frame (points, boxes) to target agent's coordinate system"""

def temporal_alignment(
    frames: List[Frame],
    target_timestamp: float,
    method: str = "nearest"  # "nearest", "interpolate"
) -> List[Frame]:
    """Align asynchronous frames to target timestamp"""

class TransformGraph:
    """Manage transformations between multiple agents"""
    
    def __init__(self):
        self.agents: Dict[str, torch.Tensor] = {}  # agent_id -> ego_pose
        
    def add_agent(self, agent_id: str, ego_pose: torch.Tensor):
        """Add agent with its global pose (4,4)"""
    
    def get_transform(self, from_agent: str, to_agent: str) -> torch.Tensor:
        """Get transformation matrix from one agent to another"""
    
    def transform_points(
        self,
        points: torch.Tensor,
        from_agent: str,
        to_agent: str
    ) -> torch.Tensor:
        """Transform points between agent coordinate frames"""
```

---

### Phase 6: Testing & Documentation

**Purpose**: Ensure library correctness and usability.

**6.1 Unit Tests** (`tests/`)

**Dataset Tests** (`tests/test_datasets.py`):

```python
def test_kitti_loader():
    """Test KITTI dataset loading and conversion"""

def test_nuscenes_sequences():
    """Test nuScenes sequence loading with tracking"""
```

**Conversion Tests** (`tests/test_converters.py`):

```python
def test_round_trip_kitti():
    """Test unified -> KITTI -> unified preserves data"""

def test_mmdet3d_format():
    """Verify MMDetection3D format structure"""
```

**Geometry Tests** (`tests/test_geometry.py`):

```python
def test_bbox_iou_3d():
    """Test 3D IoU calculation accuracy"""
```

**6.2 Documentation**


**6.3 Examples** (`examples/`)


### Phase 7: Package Setup

---

## To-dos

- [ ] Phase 1: Define unified format with tracking & temporal support, base dataset class with PyTorch
- [ ] Phase 1: Implement geometry, bbox, I/O utilities with torch operations
- [ ] Phase 2: Implement KITTI, nuScenes, Waymo, ONCE dataset loaders
- [ ] Phase 2: Implement DAIR-V2X, OPV2V cooperative dataset loaders
- [ ] Phase 3: Build framework converters (MMDetection3D, OpenPCDet)
- [ ] Phase 3: Build dataset format converters (to KITTI, nuScenes)
- [ ] Phase 4: Implement Rerun-based visualization (point clouds, BEV, camera, temporal)
- [ ] Phase 4: Build multi-agent cooperative visualization
- [ ] Phase 5: Implement coordinate transforms for cooperative detection (NO fusion)
- [ ] Phase 6: Write tests, documentation, and examples
- [ ] Phase 7: Setup package structure and dependencies
