# EN3160 Assignment 3 — Neural Networks

Implementations for EN3160 Assignment 3 (October 2026).

## Tasks

1. **CIFAR-10 manual MLP:** 3072 → 100 sigmoid → 10 logits; manual forward and backward propagation; cross-entropy; 10 epochs.
2. **MNIST CNNs:** LeNet-5 vs a 3-convolution CNN with batch normalization and dropout; 10 epochs each; accuracy/loss curves and confusion matrices.
3. **Hymenoptera transfer learning:** ImageNet-pretrained ResNet18 with full fine-tuning versus frozen-backbone feature extraction.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Question 1

```powershell
python src/q1_manual_mlp.py --epochs 10
```

CIFAR-10 is downloaded into `data/` on first execution. Training and test metrics are written to `results/q1/metrics.csv`. GPU is used if available. Do not claim training accuracy until the experiment has been executed.

## Submission

Prepare an executed Jupyter Notebook and export it as PDF, with actual results, code, comparisons and interpretation. The assignment PDF calls the submission filename `your_index_a02.pdf` despite being Assignment 3; confirm the intended name before submitting.
