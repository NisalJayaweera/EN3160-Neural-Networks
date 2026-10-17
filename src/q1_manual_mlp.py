"""Q1: CIFAR-10 MLP with explicit manual gradients, 100 sigmoid hidden units."""
import argparse
import csv
from pathlib import Path

import torch
from torchvision import datasets, transforms


def evaluate(loader, w1, b1, w2, b2, device):
    correct = total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for images, labels in loader:
            x, labels = images.to(device).flatten(1), labels.to(device)
            logits = torch.sigmoid(x @ w1 + b1) @ w2 + b2
            loss_sum += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
    return loss_sum / total, 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(args.data_dir, train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    train_eval = torch.utils.data.DataLoader(trainset, batch_size=500, shuffle=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=500, shuffle=False)

    # Parameters are tensors, not nn.Parameters: all gradients are derived manually.
    w1 = torch.randn(3072, 100, device=device) * (1 / 3072**0.5)
    b1 = torch.zeros(100, device=device)
    w2 = torch.randn(100, 10, device=device) * (1 / 100**0.5)
    b2 = torch.zeros(10, device=device)
    output = Path("results/q1")
    output.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(args.epochs):
        for images, labels in trainloader:
            x, labels = images.to(device).flatten(1), labels.to(device)
            hidden = torch.sigmoid(x @ w1 + b1)
            logits = hidden @ w2 + b2
            probs = torch.softmax(logits, dim=1)
            # For mean cross entropy with softmax: dL/dlogits = (p - onehot)/N.
            dlogits = (probs - torch.nn.functional.one_hot(labels, 10)) / len(labels)
            dw2 = hidden.T @ dlogits
            db2 = dlogits.sum(0)
            dhidden = (dlogits @ w2.T) * hidden * (1 - hidden)
            dw1 = x.T @ dhidden
            db1 = dhidden.sum(0)
            w1 -= args.lr * dw1
            b1 -= args.lr * db1
            w2 -= args.lr * dw2
            b2 -= args.lr * db2

        train_loss, train_acc = evaluate(train_eval, w1, b1, w2, b2, device)
        test_loss, test_acc = evaluate(testloader, w1, b1, w2, b2, device)
        history.append((epoch + 1, train_loss, train_acc, test_loss, test_acc))
        print(f"Epoch {epoch+1:02d}: train {train_acc:.2f}% / {train_loss:.4f}; "
              f"test {test_acc:.2f}% / {test_loss:.4f}")

    with (output / "metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "test_loss", "test_accuracy"])
        writer.writerows(history)


if __name__ == "__main__":
    main()
