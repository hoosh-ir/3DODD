from typing import Tuple, List
from core.unified_format import BBox3D  # اگه اسم فایل فرق داره اصلاحش کن

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
        # TODO: Add mappings for other datasets (waymo, once, dair_v2x, opv2v)
    }
    
    def __init__(self):
        # Map between unified class names <-> IDs
        self.class_to_id = {name: idx for idx, name in enumerate(self.UNIFIED_CLASSES)}
        self.id_to_class = {idx: name for name, idx in self.class_to_id.items()}
    
    # -------------------------------------------------------------------------
    def map_dataset_class_to_unified(
        self, 
        dataset_name: str, 
        dataset_class: str
    ) -> Tuple[str, int]:
        """Map dataset-specific class name to unified class name and ID.
        
        Example:
            ("kitti", "Van") → ("Car", 0)
        """
        dataset_name = dataset_name.lower()
        mapping = self.DATASET_MAPPINGS.get(dataset_name, {})
        
        unified_name = mapping.get(dataset_class, "Background")
        
        unified_id = self.class_to_id.get(unified_name, self.class_to_id["Background"])
        
        return unified_name, unified_id

    def get_unified_id(self, unified_class_name: str) -> int:
        """Get unified class ID from unified class name"""
        if unified_class_name not in self.class_to_id:
            raise ValueError(f"Unknown unified class name: {unified_class_name}")
        return self.class_to_id[unified_class_name]

    def filter_classes(
        self, 
        bboxes: List[BBox3D], 
        allowed_classes: List[str]
    ) -> List[BBox3D]:
        """Filter bboxes to only include specified unified classes"""
        allowed_set = set(allowed_classes)
        return [bbox for bbox in bboxes if bbox.class_name in allowed_set]
