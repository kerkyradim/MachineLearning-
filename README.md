# Machine Learning Course Projects (Harokopio University)

**Author:** [Kerkyra Dimisianou](https://github.com/kerkyradim) · 
**Semester:** 7th — Department of Informatics and Telematics  
**Repository:** [kerkyradim/MachineLearning-](https://github.com/kerkyradim/MachineLearning-)

Coursework from the **Machine Learning** module: regression, **dermoscopy image classification**, and NLP. 

---

## Featured project: dermoscopy classification (Assignment 2)

**Goal:** classify dermatoscopic skin lesions from the course **dermoscopy** image set using deep learning.

| Aspect | Details |
|--------|---------|
| **Data** | Image folders + `metadata.csv` (train/val/test split, custom `Dataset` loader) |
| **Framework** | **PyTorch** (`torch`, `torchvision`) |
| **Models** | Custom CNNs (`Net1`, `Net2` with `Conv2d` stacks) and **ResNet-34** transfer learning |
| **Training** | Cross-entropy loss, Adam/SGD, accuracy & loss curves, evaluation on held-out test set |
| **Artifacts** | [`dermoscopy_cnn_classification.ipynb`](assignment-02-dermoscopy-classification/dermoscopy_cnn_classification.ipynb) · [`ML_2nd.pdf`](assignment-02-dermoscopy-classification/ML_2nd.pdf) (report) |

The notebook expects the dataset archive `dermoscopy_classification.tar.gz` (not included in this repo due to size). Place extracted data as described in the notebook (`metadata.csv` + image directories).

### Quick start (Assignment 2)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
jupyter lab assignment-02-dermoscopy-classification/dermoscopy_cnn_classification.ipynb
```

---

## Other assignments in this repo

| Folder | Topic |
|--------|--------|
| [`assignment-01-linear-regression/`](assignment-01-linear-regression/) | Linear regression (`linear_regression.py`, tests) |
| [`assignment-03-gpt2/`](assignment-03-gpt2/) | GPT-2 fine-tuning / NLP deliverable (`gpt2_finetuning.ipynb`, report PDF) |


---

