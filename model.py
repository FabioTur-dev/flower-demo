#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os


# =========================================================
#  DEVICE SETUP
# =========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
#  GPU INFO
# =========================================================
def print_gpu_info():
    print("===== GPU INFO =====")
    print("torch version:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("Current device index:", torch.cuda.current_device())
        print("GPU name:", torch.cuda.get_device_name(0))
        print("Capability:", torch.cuda.get_device_capability(0))
    print("=====================\n")


# =========================================================
#  MODELS
# =========================================================

# ----- LeNet5 for MNIST (1-channel) -----
class LeNet5_MNIST(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # 28→24
        x = F.max_pool2d(x, 2)      # 24→12
        x = F.relu(self.conv2(x))   # 12→8
        x = F.max_pool2d(x, 2)      # 8→4
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ----- LeNet5 for CIFAR10 (3-channel) -----
class LeNet5_CIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # CIFAR 32x32 → final 5×5
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # 32→28
        x = F.max_pool2d(x, 2)      # 28→14
        x = F.relu(self.conv2(x))   # 14→10
        x = F.max_pool2d(x, 2)      # 10→5
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# =========================================================
#  DATASET LOADERS
# =========================================================

def load_mnist(batch=32):
    transform = transforms.ToTensor()
    trainset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    testset  = datasets.MNIST("./data", train=False, download=True, transform=transform)
    return (
        DataLoader(trainset, batch_size=batch, shuffle=True),
        DataLoader(testset, batch_size=batch),
    )


def load_cifar10(batch=32):
    transform = transforms.ToTensor()
    trainset = datasets.CIFAR10("./data", train=True, download=True, transform=transform)
    testset  = datasets.CIFAR10("./data", train=False, download=True, transform=transform)
    return (
        DataLoader(trainset, batch_size=batch, shuffle=True),
        DataLoader(testset, batch_size=batch),
    )


# =========================================================
#  VISUAL DEBUGGING OF DATA
# =========================================================

def show_debug_images(dataloader, dataset_name, n=6):
    print(f"\nShowing debug images from {dataset_name}...\n")
    images, labels = next(iter(dataloader))
    images, labels = images[:n], labels[:n]

    fig, axes = plt.subplots(1, n, figsize=(12, 3))

    for i in range(n):
        img = images[i].permute(1, 2, 0)

        if img.shape[-1] == 1:  # MNIST
            img = img.squeeze(-1)
            axes[i].imshow(img, cmap="gray")
        else:                    # CIFAR-10
            axes[i].imshow(img)

        axes[i].set_title(f"{labels[i].item()}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


# =========================================================
#  CENTRALIZED TRAINING
# =========================================================

def train_centralized(model, trainloader, testloader, epochs=30, device="cpu", out_csv="results/centralized.csv"):

    os.makedirs("results", exist_ok=True)
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    acc_list = []

    for epoch in range(epochs):
        model.train()
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

        # Evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)

        acc = correct / total
        acc_list.append(acc)
        print(f"[Centralized] Epoch {epoch+1} — Acc={acc:.4f}")

    # Save
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "accuracy"])
        for i, acc in enumerate(acc_list):
            w.writerow([i+1, acc])


# =========================================================
#  STANDALONE TRAINING
# =========================================================

def train_standalone(model, trainloader, testloader, epochs=10, device="cpu", out_csv="results/standalone.csv"):

    os.makedirs("results", exist_ok=True)
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    acc_list = []

    for epoch in range(epochs):
        model.train()
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

        # Evaluation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)

        acc = correct / total
        acc_list.append(acc)
        print(f"[Standalone] Epoch {epoch+1} — Acc={acc:.4f}")

    # Save
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "accuracy"])
        for i, acc in enumerate(acc_list):
            w.writerow([i+1, acc])


# =========================================================
#  MAIN
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10"], default="mnist")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--centralized", action="store_true")
    parser.add_argument("--standalone", action="store_true")
    args = parser.parse_args()

    print_gpu_info()

    # Load dataset
    if args.dataset == "mnist":
        model = LeNet5_MNIST().to(device)
        trainloader, testloader = load_mnist(args.batch)
        dataset_name = "MNIST"
    else:
        model = LeNet5_CIFAR().to(device)
        trainloader, testloader = load_cifar10(args.batch)
        dataset_name = "CIFAR-10"

    print(f"Loaded dataset: {dataset_name}")

    # Debug: show sample images
    show_debug_images(trainloader, dataset_name)

    # Training modes
    if args.centralized:
        train_centralized(model, trainloader, testloader, epochs=10, device=device)

    elif args.standalone:
        subset_size = len(trainloader.dataset) // 10
        subset, _ = torch.utils.data.random_split(trainloader.dataset,
                                                  [subset_size, len(trainloader.dataset) - subset_size])
        subset_loader = torch.utils.data.DataLoader(subset, batch_size=args.batch, shuffle=True)

        train_standalone(model, subset_loader, testloader, epochs=10, device=device)

    else:
        print("\nNo training mode selected. Use:")
        print("  --centralized   → centralized training")
        print("  --standalone    → standalone client training")


if __name__ == "__main__":
    main()

