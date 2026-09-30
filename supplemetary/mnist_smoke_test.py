import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# --- Optional: sanity-check the other required libraries import cleanly ---
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend for smoke test
import matplotlib.pyplot as plt
import cv2
import monai

print("=" * 60)
print("ENVIRONMENT CHECK")
print("=" * 60)
print(f"PyTorch version     : {torch.__version__}")
print(f"CUDA available       : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device            : {torch.cuda.get_device_name(0)}")
print(f"NumPy version        : {np.__version__}")
print(f"Pandas version        : {pd.__version__}")
print(f"OpenCV version        : {cv2.__version__}")
print(f"MONAI version         : {monai.__version__}")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}\n")


# --- A tiny CNN, just enough to prove training works end-to-end ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 28x28 -> 14x14
        x = self.pool(F.relu(self.conv2(x)))   # 14x14 -> 7x7
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_dataloaders(data_root="./data", train_subset_size=2000, test_subset_size=500, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_full = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_full = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    # Use small subsets so this stays a *smoke test* (fast), not a full training run.
    train_subset = Subset(train_full, range(train_subset_size))
    test_subset = Subset(test_full, range(test_subset_size))

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    start = time.time()

    train_loader, test_loader = get_dataloaders()

    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print("Training for 1 epoch on a 2,000-image MNIST subset...")
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Train loss: {train_loss:.4f}")

    acc = evaluate(model, test_loader, device)
    print(f"Test accuracy on 500-image subset: {acc * 100:.2f}%")

    elapsed = time.time() - start
    print(f"\nTotal smoke test time: {elapsed:.1f}s")

    # Quick OpenCV + Matplotlib sanity check: grab one image, resize with cv2,
    # save a plot. This confirms cv2 and matplotlib aren't just importable but
    # actually work on real tensor->numpy data.
    sample_img, sample_label = train_loader.dataset[0]
    img_np = (sample_img.squeeze().numpy() * 0.3081 + 0.1307)  # unnormalize
    img_np = (img_np * 255).astype(np.uint8)
    resized = cv2.resize(img_np, (56, 56), interpolation=cv2.INTER_LINEAR)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(img_np, cmap="gray")
    axes[0].set_title(f"Original (label={sample_label})")
    axes[1].imshow(resized, cmap="gray")
    axes[1].set_title("Resized w/ OpenCV")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("smoke_test_sample.png")
    print("Saved sample visualization to smoke_test_sample.png")

    if acc > 0.5:
        print("\n SMOKE TEST PASSED — environment is working end-to-end.")
    else:
        print("\n Training ran, but accuracy is surprisingly low. Check setup.")


if __name__ == "__main__":
    main()