import numpy as np
from abc import ABC , abstractmethod
from typing import Optional , List
from core.unified_format import Sample

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