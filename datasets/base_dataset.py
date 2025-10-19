import os
import json
import time
import pickle
import hashlib
import torch
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional, Callable
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from core.unified_format import Sample
# from core.collate import collate_3d_samples
from core.class_registry import ClassRegistry


class Base3DDataset(Dataset):
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
        super().__init__()

        # Dataset configuration
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.load_images = load_images
        self.lazy_load_images = lazy_load_images
        self.sequence_length = sequence_length
        self.modalities = modalities
        self.target_frame = target_frame
        self.use_pickle = use_pickle
        self.auto_create_cache = auto_create_cache
        self.cache_version_check = cache_version_check
        self.class_registry = class_registry or ClassRegistry()

        # Pickle cache setup
        default_cache_dir = os.path.join(self.data_root, "cache")
        os.makedirs(default_cache_dir, exist_ok=True)
        self.pickle_path = pickle_path or os.path.join(default_cache_dir, f"{split}.pkl")

        # Load sample list or pickle cache
        if self.use_pickle:
            if not os.path.exists(self.pickle_path):
                if self.auto_create_cache:
                    print(f"[INFO] Pickle cache not found. Creating new cache at {self.pickle_path}")
                    self.create_pickle_cache(self.pickle_path)
                else:
                    raise FileNotFoundError(f"[ERROR] Pickle cache not found at {self.pickle_path}")
            else:
                self.load_pickle_cache(self.pickle_path)
        else:
            self.sample_list = self._load_sample_list()

    # -------------------------------------------------------------------------
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.sample_list)

    # -------------------------------------------------------------------------
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
        if self.use_pickle and hasattr(self, "samples"):
            sample = self.samples[idx]
        else:
            raw_data = self.load_raw_data(idx)
            sample = self.to_unified(raw_data)

        # Apply transformation if available
        if self.transform is not None:
            sample = self.transform(sample)

        # Optional image lazy-loading logic (placeholder)
        if self.lazy_load_images and sample.data.images is None and self.load_images:
            # Future extension: implement on-demand image loading
            pass

        return sample

    # -------------------------------------------------------------------------
    def create_pickle_cache(
        self,
        output_path: Optional[str] = None,
        num_workers: int = 4,
        show_progress: bool = True,
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
        output_path = output_path or self.pickle_path
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        sample_list = self._load_sample_list()
        samples = []

        print(f"[INFO] Creating pickle cache with {len(sample_list)} samples...")

        if num_workers > 1:
            # Parallel processing using ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                iterator = executor.map(lambda i: self.to_unified(self.load_raw_data(i)), range(len(sample_list)))
                if show_progress:
                    iterator = tqdm(iterator, total=len(sample_list))
                samples = list(iterator)
        else:
            # Sequential processing
            for i in tqdm(range(len(sample_list)), disable=not show_progress):
                raw = self.load_raw_data(i)
                unified = self.to_unified(raw)
                samples.append(unified)

        # Create cache metadata
        cache_metadata = {
            "version": self.DATASET_VERSION,
            "timestamp": time.time(),
            "config_hash": self._compute_config_hash(),
            "num_samples": len(samples),
        }

        # Save to pickle
        with open(output_path, "wb") as f:
            pickle.dump({"metadata": cache_metadata, "samples": samples}, f)

        print(f"[INFO] Pickle cache successfully saved at {output_path}")

    # -------------------------------------------------------------------------
    def load_pickle_cache(self, pickle_path: Optional[str] = None) -> None:
        """Load preprocessed samples from pickle file
        
        Validates:
        - Cache version matches DATASET_VERSION
        - Configuration hash matches current settings
        - Source data not modified after cache creation (if cache_version_check=True)
        
        Raises:
            ValueError: If cache is invalid or outdated
        """
        pickle_path = pickle_path or self.pickle_path
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"[ERROR] Pickle file not found: {pickle_path}")

        with open(pickle_path, "rb") as f:
            data = pickle.load(f)

        metadata = data["metadata"]
        if self.cache_version_check and not self._validate_cache(metadata):
            raise ValueError("[ERROR] Pickle cache invalid or outdated. Please recreate it.")

        self.samples = data["samples"]
        self.sample_list = list(range(len(self.samples)))

        print(f"[INFO] Loaded {len(self.samples)} samples from pickle cache: {pickle_path}")

    # -------------------------------------------------------------------------
    def _validate_cache(self, cache_metadata: Dict[str, Any]) -> bool:
        """Validate pickle cache is compatible and up-to-date
        
        Returns:
            True if cache is valid, False otherwise
        """
        # Check dataset version
        if cache_metadata.get("version") != self.DATASET_VERSION:
            print("[WARN] Cache version mismatch — invalidating cache.")
            return False

        # Check configuration hash consistency
        current_hash = self._compute_config_hash()
        if cache_metadata.get("config_hash") != current_hash:
            print("[WARN] Dataset configuration hash mismatch — invalidating cache.")
            return False

        return True

    # -------------------------------------------------------------------------
    def _compute_config_hash(self) -> str:
        """Compute hash of dataset configuration for cache validation"""
        cfg = {
            "data_root": self.data_root,
            "split": self.split,
            "modalities": sorted(self.modalities),
            "target_frame": self.target_frame,
        }
        cfg_str = json.dumps(cfg, sort_keys=True)
        return hashlib.sha256(cfg_str.encode("utf-8")).hexdigest()

    # -------------------------------------------------------------------------
    @staticmethod
    def get_dataloader_config(
        batch_size: int = 1,
        num_workers: int = 4,
        use_pickle: bool = False,
    ) -> Dict[str, Any]:
        """Get recommended DataLoader configuration
        
        Returns:
            Dict with DataLoader kwargs optimized for this dataset
        """
        return {
            "batch_size": batch_size,
            "num_workers": 2 if use_pickle else num_workers,
            "pin_memory": True,
            # "collate_fn": collate_3d_samples,
            "persistent_workers": num_workers > 0,
            "prefetch_factor": 2 if num_workers > 0 else None,
        }
