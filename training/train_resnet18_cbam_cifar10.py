# train_resnet18_cbam_cifar10.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

from models.resnet18_cbam_model import ResNet18_CBAM


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # CIFAR-10: 3 channels, 32x32 images
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    model = ResNet18_CBAM(in_channels=3, num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                           weight_decay=5e-4, nesterov=True)
    num_epochs = 50
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_acc = 0.0

    for epoch in range(num_epochs):
        # ---- Training ----
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

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

        # ---- Validation ----
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_loss = val_loss / val_total
        val_acc = 100.0 * val_correct / val_total

        scheduler.step()

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "resnet18_cbam_cifar10_best.pth")

    torch.save(model.state_dict(), "resnet18_cbam_cifar10_final.pth")
    print(f"Training complete. Best Val Acc: {best_acc:.2f}%")
    print("Models saved: resnet18_cbam_cifar10_best.pth, resnet18_cbam_cifar10_final.pth")


if __name__ == "__main__":
    main()# train_resnet18_cbam_cifar10.py

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
    os.makedirs("./checkpoints/resnet18_cbam/cifar10", exist_ok=True)
    os.makedirs("./results/resnet18_cbam/cifar10", exist_ok=True)

    # ---------------------------------------------------
    # Data transforms
    # ---------------------------------------------------
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2470, 0.2435, 0.2616)
        )
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./datasets/data",
        train=True,
        download=True,
        transform=transform_train
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root="./datasets/data",
        train=False,
        download=True,
        transform=transform_test
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
        in_channels=3,
        num_classes=10
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True
    )

    num_epochs = 50

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs
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
                "./checkpoints/resnet18_cbam/cifar10/best.pth"
            )

    # ---------------------------------------------------
    # Save final model
    # ---------------------------------------------------

    torch.save(
        model.state_dict(),
        "./checkpoints/resnet18_cbam/cifar10/final.pth"
    )

    # ---------------------------------------------------
    # Save training history
    # ---------------------------------------------------

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        "./results/resnet18_cbam/cifar10/history.csv",
        index=False
    )

    print(f"\nTraining Complete!")
    print(f"Best Validation Accuracy : {best_acc:.2f}%")
    print("Best Model  : checkpoints/resnet18_cbam/cifar10/best.pth")
    print("Final Model : checkpoints/resnet18_cbam/cifar10/final.pth")
    print("History CSV : results/resnet18_cbam/cifar10/history.csv")


if __name__ == "__main__":
    main()