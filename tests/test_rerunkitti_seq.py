import sys
import os
import time
import rerun as rr

# Add the project root to Python path
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

    visualizer = RerunVisualizer(recording_name="KITTI_3DODD_Sequential")

    entity_path = "/kitti/sequence"

    for idx in range(min(50, len(dataset))):
        sample = dataset[idx]

        rr.set_time_sequence("frame", idx)

        visualizer.log_sample(
            sample,
            entity_path=entity_path,
            color_mode="height"
        )

        print(f"✅ Logged frame {idx}: {sample.data.frame_id}")
        time.sleep(0.03)

    print("🎬 Finished logging all frames sequentially!")


if __name__ == "__main__":
    main()
