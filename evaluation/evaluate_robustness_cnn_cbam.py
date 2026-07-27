# evaluate_robustness_resnet18_cbam.py

import os
import argparse

import torch

from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

import pandas as pd
import matplotlib.pyplot as plt

from models.resnet18_cbam_model import ResNet18_CBAM



# ---------------------------------------------------
# Transform wrappers
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
# Evaluation function
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
# Dataset loading
# ---------------------------------------------------

def get_dataset(dataset_name, extra_transform):


    if dataset_name == "cifar10":


        transform = transforms.Compose([

            extra_transform,

            transforms.ToTensor(),

            transforms.Normalize(

                (0.4914, 0.4822, 0.4465),

                (0.2470, 0.2435, 0.2616)

            )

        ])


        dataset = torchvision.datasets.CIFAR10(

            root="./datasets/data",

            train=False,

            download=True,

            transform=transform

        )


        in_channels = 3

        img_size = 32



    elif dataset_name == "mnist":


        transform = transforms.Compose([

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

            transform=transform

        )


        in_channels = 1

        img_size = 28



    else:

        raise ValueError(
            "Dataset must be cifar10 or mnist"
        )


    return dataset, in_channels, img_size




# ---------------------------------------------------
# Load model
# ---------------------------------------------------

def load_model(dataset_name, checkpoint_path, device):


    if dataset_name == "cifar10":


        model = ResNet18_CBAM(

            in_channels=3,

            num_classes=10

        )


    else:


        model = ResNet18_CBAM(

            in_channels=1,

            num_classes=10

        )


    checkpoint = torch.load(

        checkpoint_path,

        map_location=device

    )


    model.load_state_dict(checkpoint)


    model.to(device)


    model.eval()


    return model




# ---------------------------------------------------
# Robustness testing
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



    # ---------------------------------------------------
    # Baseline
    # ---------------------------------------------------

    dataset,_,_ = get_dataset(

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



    # ---------------------------------------------------
    # Translation
    # ---------------------------------------------------

    translation_values = list(range(2,42,2))


    for px in translation_values:


        dataset,_,_ = get_dataset(

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



    # ---------------------------------------------------
    # Rotation
    # ---------------------------------------------------

    rotation_values = [

        2,5,8,10,12,

        15,18,20,

        25,30,35,

        40,45,50

    ]


    for angle in rotation_values:


        dataset,_,_ = get_dataset(

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



    # ---------------------------------------------------
    # Flips
    # ---------------------------------------------------

    for name, transform in [

        ("h_flip", HFlipTransform()),

        ("v_flip", VFlipTransform())

    ]:


        dataset,_,_ = get_dataset(

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



    # ---------------------------------------------------
    # Save CSV
    # ---------------------------------------------------

    df = pd.DataFrame(results)


    csv_path = (

        f"./results/resnet18_cbam/{dataset_name}/robustness.csv"

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

        f"ResNet18+CBAM {dataset_name.upper()} Translation"

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

        f"ResNet18+CBAM {dataset_name.upper()} Rotation"

    )


    axes[1].set_xlabel(

        "Degrees"

    )


    axes[1].set_ylabel(

        "Accuracy (%)"

    )



    plt.tight_layout()



    path = (

        f"./plots/resnet18_cbam/{dataset_name}/robustness.png"

    )


    plt.savefig(

        path,

        dpi=150

    )


    print(

        f"Plot saved to {path}"

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