import os
from torchvision.datasets import Kinetics400
from torchvision import transforms
import shutil

def save_videos_to_folders(split, out_root, num_classes=8):
    ds = Kinetics400(
        root="kinetics_data",
        split=split,
        download=True,
        num_classes=num_classes,
        transform=None
    )

    print(f"Total videos in {split}: {len(ds)}")

    for idx in range(len(ds)):
        video, label = ds[idx]  # video is a tensor (T,C,H,W)
        class_name = ds.classes[label]
        class_dir = os.path.join(out_root, split, class_name)
        os.makedirs(class_dir, exist_ok=True)

        video_path = os.path.join(class_dir, f"{idx}.mp4")
        # TorchVision stores original file path
        src = ds.video_clips.video_paths[idx]

        try:
            shutil.copy(src, video_path)
        except:
            pass

    print(f"Saved videos for split={split} into {out_root}")

if __name__ == "__main__":
    out_root = "../dataset/KineticsMini"
    save_videos_to_folders("train", out_root, num_classes=8)
    save_videos_to_folders("val", out_root, num_classes=8)
