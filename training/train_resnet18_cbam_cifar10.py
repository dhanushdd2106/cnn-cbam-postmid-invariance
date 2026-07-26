# train_resnet18_cbam_cifar10.py

import os
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms

import pandas as pd

from datasets import load_dataset

from models.resnet18_cbam_model import ResNet18_CBAM


# ---------------------------------------------------
# Hugging Face CIFAR-10 Dataset Wrapper
# ---------------------------------------------------
class CIFAR10HFDataset(Dataset):

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        image = self.dataset[idx]["img"]
        label = self.dataset[idx]["label"]

        if self.transform:
            image = self.transform(image)

        return image, label


# ---------------------------------------------------
# Main Training Function
# ---------------------------------------------------
def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")


    # ---------------------------------------------------
    # Create directories
    # ---------------------------------------------------

    os.makedirs(
        "./checkpoints/resnet18_cbam/cifar10",
        exist_ok=True
    )

    os.makedirs(
        "./results/resnet18_cbam/cifar10",
        exist_ok=True
    )


    # ---------------------------------------------------
    # Data Transforms
    # ---------------------------------------------------

    transform_train = transforms.Compose([

        transforms.RandomCrop(
            32,
            padding=4
        ),

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



    # ---------------------------------------------------
    # Load CIFAR-10 from Hugging Face
    # ---------------------------------------------------

    print("Loading CIFAR-10 from Hugging Face...")

    hf_dataset = load_dataset(
        "uoft-cs/cifar10"
    )


    train_dataset = CIFAR10HFDataset(
        hf_dataset["train"],
        transform_train
    )


    test_dataset = CIFAR10HFDataset(
        hf_dataset["test"],
        transform_test
    )


    print(
        f"Training samples: {len(train_dataset)}"
    )

    print(
        f"Testing samples: {len(test_dataset)}"
    )


    # ---------------------------------------------------
    # Data Loaders
    # ---------------------------------------------------

    train_loader = DataLoader(

        train_dataset,

        batch_size=128,

        shuffle=True,

        num_workers=2,

        pin_memory=True
    )


    test_loader = DataLoader(

        test_dataset,

        batch_size=128,

        shuffle=False,

        num_workers=2,

        pin_memory=True
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


    num_epochs = 25


    scheduler = optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=num_epochs

    )


    best_acc = 0.0

    history = []


    # ---------------------------------------------------
    # Training Loop
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


            loss = criterion(
                outputs,
                labels
            )


            loss.backward()


            optimizer.step()



            running_loss += (
                loss.item()
                *
                images.size(0)
            )


            _, predicted = outputs.max(1)


            correct += (
                predicted.eq(labels)
                .sum()
                .item()
            )


            total += labels.size(0)



        train_loss = running_loss / total

        train_acc = (
            100.0 *
            correct /
            total
        )



        # ---------------------------------------------------
        # Validation
        # ---------------------------------------------------

        model.eval()


        val_loss = 0.0

        val_correct = 0

        val_total = 0



        with torch.no_grad():


            for images, labels in test_loader:


                images = images.to(device)

                labels = labels.to(device)



                outputs = model(images)



                loss = criterion(
                    outputs,
                    labels
                )



                val_loss += (
                    loss.item()
                    *
                    images.size(0)
                )


                _, predicted = outputs.max(1)


                val_correct += (
                    predicted.eq(labels)
                    .sum()
                    .item()
                )


                val_total += labels.size(0)



        val_loss = val_loss / val_total


        val_acc = (
            100.0 *
            val_correct /
            val_total
        )



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



        # Save best model

        if val_acc > best_acc:


            best_acc = val_acc


            torch.save(

                model.state_dict(),

                "./checkpoints/resnet18_cbam/cifar10/best.pth"

            )



    # ---------------------------------------------------
    # Save Final Model
    # ---------------------------------------------------

    torch.save(

        model.state_dict(),

        "./checkpoints/resnet18_cbam/cifar10/final.pth"

    )



    # ---------------------------------------------------
    # Save Training History
    # ---------------------------------------------------

    history_df = pd.DataFrame(history)


    history_df.to_csv(

        "./results/resnet18_cbam/cifar10/history.csv",

        index=False

    )



    print("\nTraining Complete")

    print(
        f"Best Validation Accuracy: {best_acc:.2f}%"
    )

    print(
        "Best Model: checkpoints/resnet18_cbam/cifar10/best.pth"
    )

    print(
        "Final Model: checkpoints/resnet18_cbam/cifar10/final.pth"
    )

    print(
        "History: results/resnet18_cbam/cifar10/history.csv"
    )



if __name__ == "__main__":

    main()