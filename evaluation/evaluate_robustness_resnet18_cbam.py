# evaluate_robustness_resnet18_cbam.py
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

from models.resnet18_cbam_model import ResNet18_CBAM


# ---------------------------------------------------
# Custom transform wrappers (deterministic, not random)
# ---------------------------------------------------
class TranslateTransform:
    """Shift image by a fixed number of pixels (both x and y)."""
    def __init__(self, px):
        self.px = px

    def __call__(self, img):
        return TF.affine(img, angle=0, translate=(self.px, self.px), scale=1.0, shear=0)


class RotateTransform:
    """Rotate image by a fixed angle (degrees)."""
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        return TF.rotate(img, angle=self.angle)


class HFlipTransform:
    def __call__(self, img):
        return TF.hflip(img)


class VFlipTransform:
    def __call__(self, img):
        return TF.vflip(img)


# ---------------------------------------------------
# Evaluation function
# ---------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def get_dataset(dataset_name, extra_transform):
    """Build test dataset with base normalization + extra transform applied AFTER ToTensor."""
    if dataset_name == "cifar10":
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            extra_transform,
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        dataset = torchvision.datasets.CIFAR10(
            root="./datasets/data", train=False, download=True, transform=base_transform
        )
        in_channels = 3

    elif dataset_name == "mnist":
        base_transform = transforms.Compose([
            transforms.ToTensor(),
            extra_transform,
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset = torchvision.datasets.MNIST(
            root="./datasets/data", train=False, download=True, transform=base_transform
        )
        in_channels = 1

    else:
        raise ValueError("dataset_name must be 'cifar10' or 'mnist'")

    return dataset, in_channels


def load_model(dataset_name, checkpoint_path, device):
    in_channels = 3 if dataset_name == "cifar10" else 1
    model = ResNet18_CBAM(in_channels=in_channels, num_classes=10)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------
# Main robustness test
# ---------------------------------------------------
def run_robustness_tests(dataset_name, checkpoint_path, batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing ResNet18+CBAM on {dataset_name.upper()} | Device: {device}")
    
     # Create required directories
    os.makedirs(f"./results/resnet18_cbam/{dataset_name}", exist_ok=True)
    os.makedirs(f"./plots/resnet18_cbam/{dataset_name}", exist_ok=True)


    model = load_model(dataset_name, checkpoint_path, device)

    results = []

    # ---- Baseline (no augmentation) ----
    identity = transforms.Lambda(lambda x: x)
    dataset, _ = get_dataset(dataset_name, identity)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    acc = evaluate(model, loader, device)
    print(f"[Baseline] Accuracy: {acc:.2f}%")
    results.append({"transform": "baseline", "value": 0, "accuracy": acc})

    # ---- Translation: 2 to 40 px, step 2 ----
    translation_values = list(range(2, 42, 2))  # 2,4,6,...,40
    for px in translation_values:
        dataset, _ = get_dataset(dataset_name, TranslateTransform(px))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        acc = evaluate(model, loader, device)
        print(f"[Translation {px}px] Accuracy: {acc:.2f}%")
        results.append({"transform": "translation", "value": px, "accuracy": acc})

    # ---- Rotation: 2,5,8,10,12,15,18,20,...,50 ----
    rotation_values = [2, 5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50]
    for angle in rotation_values:
        dataset, _ = get_dataset(dataset_name, RotateTransform(angle))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        acc = evaluate(model, loader, device)
        print(f"[Rotation {angle} deg] Accuracy: {acc:.2f}%")
        results.append({"transform": "rotation", "value": angle, "accuracy": acc})

    # ---- Horizontal Flip ----
    dataset, _ = get_dataset(dataset_name, HFlipTransform())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    acc = evaluate(model, loader, device)
    print(f"[Horizontal Flip] Accuracy: {acc:.2f}%")
    results.append({"transform": "h_flip", "value": None, "accuracy": acc})

    # ---- Vertical Flip ----
    dataset, _ = get_dataset(dataset_name, VFlipTransform())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    acc = evaluate(model, loader, device)
    print(f"[Vertical Flip] Accuracy: {acc:.2f}%")
    results.append({"transform": "v_flip", "value": None, "accuracy": acc})

    # Save results
    # Save results
    df = pd.DataFrame(results)

    csv_name = f"./results/resnet18_cbam/{dataset_name}/robustness.csv"

    df.to_csv(csv_name, index=False)

    print(f"\nResults saved to {csv_name}")

    plot_results(df, dataset_name)

    return df


# ---------------------------------------------------
# Plotting
# ---------------------------------------------------
def plot_results(df, dataset_name):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    trans_df = df[df["transform"] == "translation"]
    axes[0].plot(trans_df["value"], trans_df["accuracy"], marker="o", color="tab:blue")
    axes[0].set_title(f"ResNet18+CBAM {dataset_name.upper()} - Accuracy vs Translation")
    axes[0].set_xlabel("Translation (pixels)")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(True)

    rot_df = df[df["transform"] == "rotation"]
    axes[1].plot(rot_df["value"], rot_df["accuracy"], marker="o", color="tab:orange")
    axes[1].set_title(f"ResNet18+CBAM {dataset_name.upper()} - Accuracy vs Rotation")
    axes[1].set_xlabel("Rotation (degrees)")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True)

    plt.tight_layout()
    plot_name = f"./plots/resnet18_cbam/{dataset_name}/robustness.png"
    plt.savefig(plot_name, dpi=150)
    print(f"Plot saved to {plot_name}")
    plt.close()


# ---------------------------------------------------
# CLI entry point
# ---------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "mnist"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained model .pth file")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    run_robustness_tests(args.dataset, args.checkpoint, args.batch_size)