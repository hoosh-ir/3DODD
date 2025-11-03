import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset3d.datasets.kitti_dataset import KITTIDataset
from dataset3d.visualization.rerun_visualizer import RerunVisualizer


def main():
    data_root = "/media/sina/New Volume/data/kitti"


    dataset = KITTIDataset(
            data_root=data_root,
            split="train",
            target_frame="lidar",
            modalities=["lidar"]
        )

    print(f"✅ Loaded dataset with {len(dataset)} samples")

    sample = dataset[44]
    print(f"✅ Loaded sample: {sample.data.frame_id}")
    visualizer = RerunVisualizer(recording_name="KITTI_3DODD_Test")
        
    visualizer.log_sample(
            sample, 
            entity_path="/kitti/frame_0", 
            color_mode="height"
        )

    print(f"✅ Frame {sample.data.frame_id} visualized successfully!")
        


if __name__ == "__main__":
    main()