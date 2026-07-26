import torch
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

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
# Custom transform wrappers
# ---------------------------------------------------

class TranslateTransform:

    def __init__(self, px):
        self.px = px

    def __call__(self, img):
        return TF.affine(
            img,
            angle=0,
            translate=(self.px, self.px),
            scale=1.0,
            shear=0
        )


class RotateTransform:

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        return TF.rotate(
            img,
            angle=self.angle
        )


class HFlipTransform:

    def __call__(self, img):
        return TF.hflip(img)


class VFlipTransform:

    def __call__(self, img):
        return TF.vflip(img)



# ---------------------------------------------------
# Evaluation
# ---------------------------------------------------

def evaluate(model, loader, device):

    model.eval()

    correct = 0
    total = 0


    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)


            outputs = model(images)

            _, predicted = outputs.max(1)


            correct += predicted.eq(labels).sum().item()

            total += labels.size(0)


    return 100.0 * correct / total



# ---------------------------------------------------
# Dataset loader
# ---------------------------------------------------

def get_dataset(dataset_name, extra_transform):


    if dataset_name == "cifar10":


        base_transform = transforms.Compose([

            extra_transform,

            transforms.ToTensor(),

            transforms.Normalize(
                (0.4914, 0.4822, 0.4465),
                (0.2470, 0.2435, 0.2616)
            )

        ])



        hf_dataset = load_dataset(
            "uoft-cs/cifar10"
        )


        dataset = CIFAR10HFDataset(

            hf_dataset["test"],

            base_transform

        )


        in_channels = 3



    elif dataset_name == "mnist":


        base_transform = transforms.Compose([

            extra_transform,

            transforms.ToTensor(),

            transforms.Normalize(
                (0.1307,),
                (0.3081,)
            )

        ])



        import torchvision


        dataset = torchvision.datasets.MNIST(

            root="./datasets/data",

            train=False,

            download=True,

            transform=base_transform

        )


        in_channels = 1



    else:

        raise ValueError(
            "dataset_name must be cifar10 or mnist"
        )


    return dataset, in_channels




# ---------------------------------------------------
# Load model
# ---------------------------------------------------

def load_model(dataset_name, checkpoint_path, device):


    in_channels = (
        3
        if dataset_name == "cifar10"
        else 1
    )


    model = ResNet18_CBAM(

        in_channels=in_channels,

        num_classes=10

    )


    model.load_state_dict(

        torch.load(
            checkpoint_path,
            map_location=device
        )

    )


    model.to(device)

    model.eval()


    return model




# ---------------------------------------------------
# Robustness testing
# ---------------------------------------------------

def run_robustness_tests(
        dataset_name,
        checkpoint_path,
        batch_size=128):


    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        f"Testing ResNet18+CBAM on {dataset_name.upper()} | Device: {device}"
    )


    os.makedirs(
        f"./results/resnet18_cbam/{dataset_name}",
        exist_ok=True
    )


    os.makedirs(
        f"./plots/resnet18_cbam/{dataset_name}",
        exist_ok=True
    )



    model = load_model(
        dataset_name,
        checkpoint_path,
        device
    )


    results = []



    # Baseline

    identity = transforms.Lambda(
        lambda x:x
    )


    dataset,_ = get_dataset(
        dataset_name,
        identity
    )


    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )


    acc = evaluate(
        model,
        loader,
        device
    )


    print(
        f"[Baseline] Accuracy: {acc:.2f}%"
    )


    results.append({

        "transform":"baseline",

        "value":0,

        "accuracy":acc

    })



    # Translation

    translation_values = list(
        range(2,42,2)
    )


    for px in translation_values:


        dataset,_ = get_dataset(

            dataset_name,

            TranslateTransform(px)

        )


        loader = DataLoader(

            dataset,

            batch_size=batch_size,

            shuffle=False,

            num_workers=2

        )


        acc = evaluate(

            model,

            loader,

            device

        )


        print(
            f"[Translation {px}px] Accuracy: {acc:.2f}%"
        )


        results.append({

            "transform":"translation",

            "value":px,

            "accuracy":acc

        })



    # Rotation

    rotation_values = [

        2,5,8,10,12,15,

        18,20,25,30,

        35,40,45,50

    ]


    for angle in rotation_values:


        dataset,_ = get_dataset(

            dataset_name,

            RotateTransform(angle)

        )


        loader = DataLoader(

            dataset,

            batch_size=batch_size,

            shuffle=False,

            num_workers=2

        )


        acc = evaluate(

            model,

            loader,

            device

        )


        print(

            f"[Rotation {angle} deg] Accuracy: {acc:.2f}%"

        )


        results.append({

            "transform":"rotation",

            "value":angle,

            "accuracy":acc

        })



    # Horizontal Flip

    for name,transform in [

        ("h_flip",HFlipTransform()),

        ("v_flip",VFlipTransform())

    ]:


        dataset,_ = get_dataset(

            dataset_name,

            transform

        )


        loader = DataLoader(

            dataset,

            batch_size=batch_size,

            shuffle=False,

            num_workers=2

        )


        acc = evaluate(

            model,

            loader,

            device

        )


        print(
            f"[{name}] Accuracy: {acc:.2f}%"
        )


        results.append({

            "transform":name,

            "value":None,

            "accuracy":acc

        })



    # Save CSV

    df = pd.DataFrame(results)


    csv_name = (

        f"./results/resnet18_cbam/"
        f"{dataset_name}/robustness.csv"

    )


    df.to_csv(
        csv_name,
        index=False
    )


    print(
        f"Results saved to {csv_name}"
    )


    plot_results(
        df,
        dataset_name
    )


    return df




# ---------------------------------------------------
# Plot
# ---------------------------------------------------

def plot_results(df, dataset_name):


    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14,5)
    )


    trans_df = df[
        df["transform"]=="translation"
    ]


    axes[0].plot(
        trans_df["value"],
        trans_df["accuracy"],
        marker="o"
    )


    axes[0].set_title(
        f"{dataset_name.upper()} Translation"
    )


    axes[0].set_xlabel(
        "Pixels"
    )


    axes[0].set_ylabel(
        "Accuracy (%)"
    )


    rot_df = df[
        df["transform"]=="rotation"
    ]


    axes[1].plot(
        rot_df["value"],
        rot_df["accuracy"],
        marker="o"
    )


    axes[1].set_title(
        f"{dataset_name.upper()} Rotation"
    )


    axes[1].set_xlabel(
        "Degrees"
    )


    axes[1].set_ylabel(
        "Accuracy (%)"
    )


    plt.tight_layout()


    plot_name = (

        f"./plots/resnet18_cbam/"
        f"{dataset_name}/robustness.png"

    )


    plt.savefig(
        plot_name,
        dpi=150
    )


    print(
        f"Plot saved to {plot_name}"
    )


    plt.close()



# ---------------------------------------------------
# Main
# ---------------------------------------------------

if __name__ == "__main__":


    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[
            "cifar10",
            "mnist"
        ]
    )


    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True
    )


    parser.add_argument(
        "--batch_size",
        type=int,
        default=128
    )


    args = parser.parse_args()



    run_robustness_tests(

        args.dataset,

        args.checkpoint,

        args.batch_size

    )