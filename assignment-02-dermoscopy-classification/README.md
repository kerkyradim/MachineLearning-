# Assignment 2 — Dermoscopy classification

**Deliverable:** `dermoscopy_cnn_classification.ipynb` · **Report:** `ML_2nd.pdf`

## Workflow

1. Obtain `dermoscopy_classification.tar.gz` from the course / ISIC bundle and extract it.
2. Point `data_dir` in the notebook to the folder containing `metadata.csv` and image subfolders.
3. Run sections in order: dataset class → dataloaders → train/test helpers → **Net1** → **Net2** → **ResNet-34** transfer learning.

## Models

- **Net1 / Net2** — convolutional stacks (`Conv2d`, pooling, fully connected head)
- **ResNet-34** — pretrained backbone fine-tuned on the dermoscopy splits

Metrics logged: training/validation **loss** and **accuracy**; final **test** evaluation with predictions for analysis.
