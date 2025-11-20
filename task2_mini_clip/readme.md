Notebook-style script: Task 2 - mini-CLIP (do NOT run; download and run on RunPod)
Author: Imran Abbas
Purpose: Implement a mini-CLIP: image encoder (ResNet-50), text encoder (Transformer), contrastive training (NT-Xent), retrieval evaluation, t-SNE visualization, zero-shot classification prompts.

How to use:
- Place this file in your RunPod workspace and open it as a notebook (or split cells manually).
- Edit CONFIG section (paths, hyperparams).
- Install dependencies: pip install torch torchvision scikit-learn matplotlib pillow tqdm

Notes:
- Dataset format expected: CSV with columns [image_path, caption]
- Image paths in CSV can be absolute or relative to `data_root`.
- The notebook code is modular: you can run individual cells.