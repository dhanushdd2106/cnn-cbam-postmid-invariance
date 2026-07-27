# evaluation/evaluate_robustness_cnn_cbam.py

import os
import argparse

import torch

from torch.utils.data import DataLoader, Dataset

import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import pandas as pd
import matplotlib.pyplot as plt

from datasets import load_dataset

from models.cnn_cbam_model import CNN_CBAM



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
# Transform classes
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
            self.angle
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
# Dataset Loader
# ---------------------------------------------------

def get_dataset(dataset_name, extra_transform):


    if dataset_name == "cifar10":


        print("Loading CIFAR-10 from Hugging Face...")


        base_transform = transforms.Compose([

        extra_transform,

        transforms.ToTensor(),

        transforms.Normalize(

            (0.4914,0.4822,0.4465),

            (0.2470,0.2435,0.2616)

        )

    ])



        hf_dataset = load_dataset(

            "uoft-cs/cifar10"

        )



        dataset = CIFAR10HFDataset(

            hf_dataset["test"],

            base_transform

        )



    elif dataset_name == "mnist":


        base_transform = transforms.Compose([

            extra_transform,

            transforms.ToTensor(),

            transforms.Normalize(

                (0.1307,),

                (0.3081,)

            )

        ])



        dataset = torchvision.datasets.MNIST(

            root="./datasets/data",

            train=False,

            download=True,

            transform=base_transform

        )


    else:

        raise ValueError(

            "dataset_name must be cifar10 or mnist"

        )


    return dataset



# ---------------------------------------------------
# Load CNN+CBAM Model
# ---------------------------------------------------

def load_model(dataset_name, checkpoint_path, device):


    if dataset_name == "cifar10":


        model = CNN_CBAM(

            in_channels=3,

            num_classes=10,

            img_size=32

        )


    else:


        model = CNN_CBAM(

            in_channels=1,

            num_classes=10,

            img_size=28

        )


    state_dict = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(state_dict)

    # --- diagnostic block: remove once issue is resolved ---
    print("Checkpoint path :", os.path.abspath(checkpoint_path))
    print("Checkpoint mtime:", os.path.getmtime(checkpoint_path))

    checksum = sum(p.sum().item() for p in model.parameters())
    print("Param checksum  :", checksum)

    num_keys = len(state_dict)
    print("State dict keys :", num_keys)
    # ---------------------------------------------------------


    model.to(device)

    model.eval()


    return model



# ---------------------------------------------------
# Robustness Testing
# ---------------------------------------------------

def run_robustness_tests(

        dataset_name,

        checkpoint_path,

        batch_size=128

):


    device = torch.device(

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )


    print(

        f"Testing CNN+CBAM on {dataset_name.upper()} | Device: {device}"

    )



    os.makedirs(

        f"./results/cnn_cbam/{dataset_name}",

        exist_ok=True

    )


    os.makedirs(

        f"./plots/cnn_cbam/{dataset_name}",

        exist_ok=True

    )



    model = load_model(

        dataset_name,

        checkpoint_path,

        device

    )


    results = []



    # ---------------- Baseline ----------------


    dataset = get_dataset(

        dataset_name,

        transforms.Lambda(lambda x:x)

    )


    loader = DataLoader(

        dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=2,

        pin_memory=True

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



    # ---------------- Translation ----------------


    translation_values = list(

        range(2,42,2)

    )


    for px in translation_values:


        dataset = get_dataset(

            dataset_name,

            TranslateTransform(px)

        )


        loader = DataLoader(

            dataset,

            batch_size=batch_size,

            shuffle=False,

            num_workers=2,

            pin_memory=True

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



    # ---------------- Rotation ----------------


    rotation_values = [

        2,5,8,10,12,

        15,18,20,

        25,30,35,

        40,45,50

    ]



    for angle in rotation_values:


        dataset = get_dataset(

            dataset_name,

            RotateTransform(angle)

        )


        loader = DataLoader(

            dataset,

            batch_size=batch_size,

            shuffle=False,

            num_workers=2,

            pin_memory=True

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



    # ---------------- Flips ----------------


    for name, transform in [

        ("h_flip", HFlipTransform()),

        ("v_flip", VFlipTransform())

    ]:


        dataset = get_dataset(

            dataset_name,

            transform

        )


        loader = DataLoader(

            dataset,

            batch_size=batch_size,

            shuffle=False,

            num_workers=2,

            pin_memory=True

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



    # ---------------- Save Results ----------------


    df = pd.DataFrame(results)


    csv_path = (

        f"./results/cnn_cbam/{dataset_name}/robustness.csv"

    )


    df.to_csv(

        csv_path,

        index=False

    )


    print(

        f"Results saved to {csv_path}"

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

        f"CNN+CBAM {dataset_name.upper()} Translation"

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

        f"CNN+CBAM {dataset_name.upper()} Rotation"

    )


    axes[1].set_xlabel(

        "Degrees"

    )


    axes[1].set_ylabel(

        "Accuracy (%)"

    )



    plt.tight_layout()



    plot_path = (

        f"./plots/cnn_cbam/{dataset_name}/robustness.png"

    )


    plt.savefig(

        plot_path,

        dpi=150

    )


    print(

        f"Plot saved to {plot_path}"

    )


    plt.close()



# ---------------------------------------------------
# Main
# ---------------------------------------------------

if __name__ == "__main__":


    parser = argparse.ArgumentParser()


    parser.add_argument(

        "--dataset",

        required=True,

        choices=[

            "cifar10",

            "mnist"

        ]

    )


    parser.add_argument(

        "--checkpoint",

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