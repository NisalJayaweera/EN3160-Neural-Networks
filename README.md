# EN3160 — Neural Networks | Assignment 03

Implementations, training results, and analysis for **EN3160: Image Processing and Machine Vision — Neural Networks Assignment 03** at the **University of Moratuwa**.

The assignment covers a multilayer perceptron implemented with manual backpropagation, convolutional neural networks for handwritten-digit recognition, and transfer learning with a pretrained ResNet18.

## Assignment overview

| Question | Dataset | Method | Main experiment |
| :--- | :--- | :--- | :--- |
| **Q1 — Manual MLP** | CIFAR-10 | 3072 → 100 sigmoid → 10 | Implement forward propagation, cross-entropy gradients, and parameter updates manually |
| **Q2 — CNN comparison** | MNIST | LeNet-5 vs. a modern 3-convolution CNN | Compare training/test accuracy and loss, plus confusion matrices |
| **Q3 — Transfer learning** | Hymenoptera (ants/bees) | ImageNet-pretrained ResNet18 | Compare full-network fine-tuning against training only the final classifier |

All three questions were trained for **10 epochs** in the saved experiments.

## Notebooks and submission

| File | Purpose |
| :--- | :--- |
| [**Final combined notebook**](notebooks/EN3160_Assignment_3_Final_Notebook.ipynb) | All three questions, code, recorded outputs, visualizations, and discussions |
| [Q1 — CIFAR-10 Manual MLP](notebooks/EN3160_Q1_CIFAR10_Manual_MLP_Backpropagation.ipynb.ipynb) | Manual backpropagation and CIFAR-10 training |
| [Q2 — MNIST CNN Comparison](notebooks/EN3160_Q2_MNIST_CNN_Comparison%20%281%29.ipynb) | LeNet-5 and modern CNN experiments |
| [Q3 — ResNet18 Transfer Learning](notebooks/EN3160_Q3_ResNet18_Transfer_Learning%20%281%29.ipynb) | Fine-tuning and feature extraction experiments |
| [**PDF report**](Final_PDF/230300C_a02.pdf) | Report stored in the repository |

> **Note:** The Q1 notebook currently has a double `.ipynb` extension, and the Q2/Q3 filenames contain `(1)`. Links above match the existing files. The PDF filename is preserved as stored; check the course submission instructions before uploading it.

[![Open the combined notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NisalJayaweera/EN3160-Neural-Networks/blob/main/notebooks/EN3160_Assignment_3_Final_Notebook.ipynb)

## Experimental results

The following values come from the **saved 10-epoch notebook outputs** (one recorded training run per experiment).

### Q1 — CIFAR-10 manual MLP

| Metric | Final value |
| :--- | ---: |
| Training accuracy | 55.46% |
| Test accuracy | 48.20% |
| Best recorded test accuracy | 48.42% |

The network uses a 100-unit sigmoid hidden layer and **explicitly computed gradients**, without PyTorch autograd for backpropagation. Because a flattened-pixel MLP does not exploit image spatial structure, performance on CIFAR-10 is limited compared with convolutional architectures.

### Q2 — LeNet-5 vs. modern CNN on MNIST

| Metric | LeNet-5 | Modern CNN |
| :--- | ---: | ---: |
| Training accuracy | 99.54% | 98.87% |
| Test accuracy | **98.76%** | 98.16% |
| Test loss | **0.0453** | 0.0541 |

Both models were evaluated on the MNIST test split. In this run, LeNet-5 achieved slightly higher accuracy despite using fewer trainable parameters. The Q2 notebook includes learning curves and test-set confusion matrices.

### Q3 — ResNet18 on Hymenoptera

| Metric | Full fine-tuning | Feature extraction |
| :--- | ---: | ---: |
| Training accuracy | 96.72% | 98.77% |
| Validation accuracy | 78.43% | **94.12%** |
| Validation loss | 0.5509 | **0.1594** |
| Training time | 26.00 s | 23.40 s |
| Trainable parameters | 11,177,538 | 1,026 |

The dataset consists of **244 training images** and **153 validation images**. Feature extraction performed better in this run, although the small dataset and single experiment do not establish that it will always outperform fine-tuning. Q3 uses the supplied validation split; it is **not** a separate held-out test split.

## Repository structure

```text
EN3160-Neural-Networks/
├── Final_PDF/
│   └── 230300C_a02.pdf
├── notebooks/
│   ├── EN3160_Assignment_3_Final_Notebook.ipynb
│   ├── EN3160_Q1_CIFAR10_Manual_MLP_Backpropagation.ipynb.ipynb
│   ├── EN3160_Q2_MNIST_CNN_Comparison (1).ipynb
│   └── EN3160_Q3_ResNet18_Transfer_Learning (1).ipynb
├── results/
├── src/
│   └── q1_manual_mlp.py
├── requirements.txt
└── README.md
```

## Running the project

### Option A — Google Colab (recommended)

1. Open the [combined notebook in Colab](https://colab.research.google.com/github/NisalJayaweera/EN3160-Neural-Networks/blob/main/notebooks/EN3160_Assignment_3_Final_Notebook.ipynb).
2. Select **Runtime → Change runtime type → GPU** if available.
3. In a fresh runtime, run cells from top to bottom. Q1 will clone the repository as needed.
4. Allow dataset and pretrained-weight downloads on the first execution. Q2 and Q3 perform training within their respective notebook cells.

**Important:** Colab ordinarily includes PyTorch and torchvision. Avoid reinstalling or upgrading them inside an already-running runtime, because this may cause import conflicts. If that occurs, disconnect/delete the runtime and start a fresh session.

The figures and metrics already displayed in the committed notebook are from previous individual runs. Re-running the **entire combined notebook** is recommended when verifying reproducibility.

### Option B — Local Python (Q1 training script)

Use Python and install the dependencies in an isolated environment:

```bash
git clone https://github.com/NisalJayaweera/EN3160-Neural-Networks.git
cd EN3160-Neural-Networks
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/q1_manual_mlp.py --epochs 10
```

The Q1 script downloads CIFAR-10 on first execution and writes per-epoch metrics to `results/q1/metrics.csv`. CUDA is used automatically when available. Use Jupyter or Colab for the Q2 and Q3 experiments.

## Methodology notes

- **Q1:** A manually differentiated MLP trained with cross-entropy on CIFAR-10.
- **Q2:** LeNet-5 and a modern CNN trained using matching MNIST splits and experimental settings.
- **Q3:** Both methods start from pretrained ResNet18 weights; fine-tuning updates the backbone, whereas feature extraction freezes it and trains the replacement classifier.
- Accuracies and training times are **observations from recorded runs**, not promises of identical results across devices, seeds, or library versions.

## References

- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
- [Torchvision datasets](https://pytorch.org/vision/stable/datasets.html)
- [PyTorch transfer learning tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

---

*Coursework repository for EN3160 — University of Moratuwa.*
