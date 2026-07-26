# train_resnet18_cbam_mnist.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import pandas as pd

from models.resnet18_cbam_model import ResNet18_CBAM


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---------------------------------------------------
    # Create required directories
    # ---------------------------------------------------
    os.makedirs("./datasets/data", exist_ok=True)
    os.makedirs("./checkpoints/resnet18_cbam/mnist", exist_ok=True)
    os.makedirs("./results/resnet18_cbam/mnist", exist_ok=True)

    # ---------------------------------------------------
    # MNIST transforms
    # ---------------------------------------------------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = torchvision.datasets.MNIST(
        root="./datasets/data",
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = torchvision.datasets.MNIST(
        root="./datasets/data",
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=2
    )

    # ---------------------------------------------------
    # Model
    # ---------------------------------------------------
    model = ResNet18_CBAM(
        in_channels=1,
        num_classes=10
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    num_epochs = 15

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.5
    )

    best_acc = 0.0
    history = []

    # ---------------------------------------------------
    # Training
    # ---------------------------------------------------
    for epoch in range(num_epochs):

        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)

            correct += predicted.eq(labels).sum().item()

            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = 100.0 * correct / total

        # ---------------- Validation ----------------

        model.eval()

        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, labels in test_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)

                _, predicted = outputs.max(1)

                val_correct += predicted.eq(labels).sum().item()

                val_total += labels.size(0)

        val_loss = val_loss / val_total
        val_acc = 100.0 * val_correct / val_total

        scheduler.step()

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_acc:

            best_acc = val_acc

            torch.save(
                model.state_dict(),
                "./checkpoints/resnet18_cbam/mnist/best.pth"
            )

    # ---------------------------------------------------
    # Save final model
    # ---------------------------------------------------
    torch.save(
        model.state_dict(),
        "./checkpoints/resnet18_cbam/mnist/final.pth"
    )

    # ---------------------------------------------------
    # Save training history
    # ---------------------------------------------------
    history_df = pd.DataFrame(history)

    history_df.to_csv(
        "./results/resnet18_cbam/mnist/history.csv",
        index=False
    )

    print("\nTraining Complete!")
    print(f"Best Validation Accuracy : {best_acc:.2f}%")
    print("Best Model  : checkpoints/resnet18_cbam/mnist/best.pth")
    print("Final Model : checkpoints/resnet18_cbam/mnist/final.pth")
    print("History CSV : results/resnet18_cbam/mnist/history.csv")


if __name__ == "__main__":
    main()