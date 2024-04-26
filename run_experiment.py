import argparse
import io
import os
import pickle
import tempfile
import time

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import yaml
from torchvision import datasets, models
import torchvision.transforms.v2 as transforms
from collections import Counter

from src.dataloader import MultiFormatDataLoader, SubsetDataset
from src.evaluator import Evaluator
from src.models import *
from src.trainer import PyTorchTrainer
from src.utils import seed_everything

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# for i in $(seq 0.1 0.1 0.5); do for j in "${arr[@]}"; do for k in "${arr2[@]}"; do sbatch run_small_"$k"_"$j".sh "$i"; done; done; done

def main(args):
    # Load the WANDB YAML file
    with open("./wandb_config.yaml") as file:
        wandb_data = yaml.load(file, Loader=yaml.FullLoader)

    os.environ["WANDB_API_KEY"] = wandb_data["wandb_key"]
    wandb_entity = wandb_data["wandb_entity"]

    total_runs = args.total_runs
    hardness = args.hardness
    dataset = args.dataset
    model_name = args.model_name
    epochs = args.epochs
    seed = args.seed
    p = args.prop

    assert dataset in ["mnist", "cifar10", "caltech256", "cifar100", "fashionmnist", "imagenet", 
    "nih", "nihpneumonia", "padchest", "vindrcxr", "objectcxr", "siim"], "Invalid dataset!"

    for i in range(total_runs):

        ####################
        #
        # SET UP EXPERIMENT
        #
        ####################

        print(f"Running {i+1}/{total_runs} for {p}")
        seed_everything(seed)
        print(f"{hardness}_{dataset}_{model_name}_{epochs}")
        dir_to_delete = None

        # new wandb run
        run = wandb.init(
            project="example_difficulty",
            entity=wandb_entity,
        )
        wandb.config({'total_runs': total_runs, 'run': i+1, 'hardness': hardness, 'dataset': dataset, 'model_name': model_name, 'total_epochs': epochs, 'seed': seed, 'prop': p})

        if hardness == "instance":
            if dataset == "mnist":
                rule_matrix = {
                    1: [7],     # 1 -> 7
                    2: [7],     # 2 -> 7
                    3: [8],     # 3 -> 8
                    4: [4],     # 4 (unchanged)
                    5: [6],     # 5 -> 6
                    6: [5],     # 6 -> 5
                    7: [1, 2],  # 7 -> 1 or 2
                    8: [3],     # 8 -> 3
                    9: [7],     # 9 -> 7
                    0: [0],     # 0 (unchanged)
                }

            if dataset == "cifar10":

                rule_matrix = {
                    0: [2],        # airplane -> bird
                    1: [9],        # automobile -> truck
                    2: [2],        # bird (unchanged)
                    3: [5],        # cat -> dog
                    4: [5, 7],     # deer -> dog or horse
                    5: [3, 4, 7],  # dog -> cat or deer or horse
                    6: [6],        # frog (unchanged)
                    7: [5],        # horse -> dog
                    8: [8],        # ship (unchanged)
                    9: [1],        # truck -> automobile
                }

            if dataset == "caltech256":

                # using caltech256.py to figure out similar items the image could be confused for
                rule_matrix = {1: [172, 173], 2: [2], 3: [8, 42, 66, 135, 165, 185, 199, 229], 4: [4], 5: [5], 6: [6], 7: [9, 28, 38, 56, 64, 65, 80, 84, 85, 90, 105, 121, 134, 164, 250, 254, 256], 8: [3, 13, 42, 66, 135, 165, 185, 199, 212, 229], 9: [7, 28, 38, 56, 64, 65, 84, 85, 90, 105, 121, 134, 164, 250, 254], 10: [10], 11: [11], 12: [67, 141, 209, 210, 219, 240], 13: [8, 66, 199, 212], 14: [14], 15: [15, 25, 68, 103], 16: [16], 17: [17], 18: [18], 19: [19], 20: [20], 21: [21], 22: [22], 23: [145, 192], 24: [34, 40, 93, 198, 230], 25: [15, 68, 118], 26: [55], 27: [27], 28: [7, 9, 56, 64, 65, 84, 85, 105, 134, 164, 250], 29: [29], 30: [122, 123], 31: [31], 32: [32], 33: [33], 34: [24, 40, 48, 52, 93, 189, 198, 201, 230], 35: [35], 36: [36], 37: [37], 38: [7, 9, 64, 90, 164], 39: [199], 40: [24, 34, 93, 198, 230], 41: [41], 42: [3, 8, 66, 135, 165, 185, 199, 229], 43: [43], 44: [44], 45: [45], 46: [46], 47: [47], 48: [34, 189, 201, 230], 49: [60, 89, 113, 114, 152, 158, 207], 50: [50], 51: [51], 52: [34, 52, 198, 230], 53: [53], 54: [54], 55: [26], 56: [7, 9, 28, 64, 84, 105, 134, 164, 250, 254], 57: [57, 87], 58: [58], 59: [59], 60: [49, 89, 113, 114, 152, 158, 207], 61: [61], 62: [62], 63: [63], 64: [7, 9, 28, 38, 56, 64, 65, 84, 85, 90, 105, 121, 134, 164, 250], 65: [7, 9, 28, 64, 84, 85, 105, 134, 164, 250], 66: [3, 8, 13, 42, 66, 135, 165, 185, 199, 212, 229], 67: [12, 141, 209, 210, 219, 240], 68: [15, 25, 118], 69: [69], 70: [70], 71: [71], 72: [72], 73: [73], 74: [131], 75: [75], 76: [76], 77: [77], 78: [78], 79: [79], 80: [7, 113, 152, 190, 207, 256], 81: [81], 82: [82], 83: [83], 84: [7, 9, 28, 56, 64, 65, 85, 105, 134, 164, 250], 85: [7, 9, 28, 64, 65, 84, 105, 134, 164, 250], 86: [86], 87: [57], 88: [88], 89: [49, 60, 113, 114, 152, 158, 207], 90: [7, 9, 38, 64, 164], 91: [91], 92: [92], 93: [24, 34, 40, 198, 230], 94: [94], 95: [206], 96: [96], 97: [247], 98: [99, 136, 247], 99: [98, 136, 247], 100: [100], 101: [101], 102: [102, 251], 103: [15], 104: [104], 105: [7, 9, 28, 56, 64, 65, 84, 85, 134, 164, 250], 106: [106], 107: [107], 108: [108], 109: [109], 110: [240], 111: [111], 112: [112], 113: [49, 60, 80, 89, 114, 152, 158, 207, 256], 114: [49, 60, 89, 113, 114, 152, 158, 207], 115: [115], 116: [190], 117: [117], 118: [25, 68], 119: [119], 120: [120], 121: [7, 9, 64, 121, 164], 122: [30, 123], 123: [30, 122, 123], 124: [124], 125: [125], 126: [126], 127: [127], 128: [128], 129: [129, 168], 130: [130], 131: [74], 132: [132], 133: [133], 134: [7, 9, 28, 56, 64, 65, 84, 85, 105, 134, 164, 250], 135: [3, 8, 42, 66, 165, 185, 199, 229], 136: [98, 99, 247], 137: [177], 138: [138], 139: [181, 203, 227], 140: [140], 141: [12, 67, 209, 210, 219], 142: [142], 143: [143], 144: [236], 145: [23, 145, 192], 146: [146], 147: [147], 148: [150], 149: [149], 150: [148], 151: [151], 152: [49, 60, 80, 89, 113, 114, 158, 207, 256], 153: [153], 154: [154], 155: [155], 156: [156], 157: [157], 158: [49, 60, 89, 113, 114, 152, 207], 159: [159], 160: [160], 161: [161], 162: [162], 163: [163], 164: [7, 9, 28, 38, 56, 64, 65, 84, 85, 90, 105, 121, 134, 250], 165: [3, 8, 42, 66, 135, 185, 199, 229, 244], 166: [166], 167: [167], 168: [129], 169: [169], 170: [170], 171: [171], 172: [172, 173, 209], 173: [172, 209], 174: [174], 175: [175], 176: [203, 231], 177: [137], 178: [178], 179: [179], 180: [234], 181: [139, 203, 227], 182: [182], 183: [183], 184: [184], 185: [3, 8, 42, 66, 135, 165, 199, 229, 244], 186: [186], 187: [187], 188: [188], 189: [34, 48, 201, 230], 190: [80, 116, 228, 256], 191: [191], 192: [23, 145], 193: [193], 194: [194], 195: [195], 196: [206], 197: [197], 198: [24, 34, 40, 52, 93, 201, 230], 199: [3, 8, 13, 39, 42, 66, 135, 165, 185, 212, 229, 244], 200: [200], 201: [34, 48, 189, 198, 201, 230], 202: [202], 203: [139, 176, 181, 227, 231], 204: [204], 205: [205], 206: [95, 196], 207: [49, 60, 80, 89, 113, 114, 152, 158, 256], 208: [208], 209: [12, 67, 141, 172, 173, 210, 219, 240], 210: [12, 67, 141, 209, 219, 240], 211: [247], 212: [8, 13, 66, 199], 213: [213], 214: [214], 215: [215], 216: [216], 217: [217], 218: [218], 219: [12, 67, 141, 209, 210, 240], 220: [220], 221: [221], 222: [222], 223: [223], 224: [224], 225: [225], 226: [226], 227: [139, 181, 203], 228: [190], 229: [3, 8, 42, 66, 135, 165, 185, 199, 244], 230: [24, 34, 40, 48, 52, 93, 189, 198, 201, 230], 231: [176, 203], 232: [232], 233: [233], 234: [180], 235: [235], 236: [144], 237: [237], 238: [238], 239: [239], 240: [12, 67, 110, 209, 210, 219, 240], 241: [241], 242: [242], 243: [243], 244: [165, 185, 199, 229], 245: [245], 246: [246], 247: [97, 98, 99, 136, 211], 248: [248], 249: [249], 250: [7, 9, 28, 56, 64, 65, 84, 85, 105, 134, 164], 251: [102, 251], 252: [252], 253: [253], 254: [7, 9, 56], 255: [255], 256: [7, 80, 113, 152, 190, 207]}
                
            if dataset == "cifar100":

                temp_matrix = {
                    0: [72, 4, 95, 30, 55],    # aquatic mammals
                    1: [73, 32, 67, 91, 1],    # fish
                    2: [92, 70, 82, 54, 62],   # flowers
                    3: [16, 61, 9, 10, 28],    # food containers
                    4: [51, 0, 53, 57, 83],    # fruit and vegetables
                    5: [40, 39, 22, 87, 86],   # household electric devices
                    6: [20, 25, 94, 84, 5],    # household furniture
                    7: [14, 24, 6, 7, 18],     # insects
                    8: [43, 97, 42, 3, 88],    # large carnivores
                    9: [37, 17, 76, 12, 68],   # large man-made outdoor things
                    10: [49, 33, 71, 23, 60],  # large natural outdoor scenes
                    11: [15, 21, 19, 31, 38],  # large omnivores and herbivores
                    12: [75, 63, 66, 64, 34],  # medium-sized mallas
                    13: [77, 26, 45, 99, 79],  # non-insect vertebrates
                    14: [11, 2, 35, 46, 98],   # people
                    15: [29, 93, 27, 78, 44],  # reptiles
                    16: [65, 50, 74, 36, 80],  # small mammals
                    17: [56, 52, 47, 59, 96],  # trees
                    18: [8, 58, 90, 13, 48],   # vehicles 1
                    19: [81, 69, 41, 89, 85]   # vehicles 2
                }

                rule_matrix = {}
                for i in temp_matrix.keys():
                    for j in range(5):
                        rule_matrix[j] = temp_matrix[i].copy().remove(j)


            if dataset == "fashionmnist":

                rule_matrix = {
                0: [2,4,6],   # t-shirt/top -> pullover or coat or shirt
                1: [1],       # trouser (unchanged)
                2: [0,4,6],   # pullover -> t-shirt/top or coat or shirt
                3: [6],       # dress -> shirt
                4: [0,2,6],   # coat -> t-shirt/top or pullover or shirt
                5: [5],       # sandal (unchanged)
                6: [0,2,3,4], # shirt -> t-shirt/top or pullover or dress or coat
                7: [9],       # sneaker -> ankle boot
                8: [8],       # bag (unchanged)
                9: [7]        # ankle boot -> sneaker
                }

        else:
            rule_matrix = None

        if dataset == "mnist":
            # Define transforms for the dataset
            # 60K, 10K, already 28x28
            transform = transforms.Compose(
                [transforms.ToTensor(), 
                transforms.Normalize((0.5,), (0.5,))]
            )
            # Load the MNIST dataset
            train_dataset = datasets.MNIST(
                root="./data", train=True, download=True, transform=transform
            )
            test_dataset = datasets.MNIST(
                root="./data", train=False, download=True, transform=transform
            )
            num_classes = 10

        elif dataset == "cifar10":
            # Define transforms for the dataset
            # 50K, 10K, already 32x32
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            # Load the CIFAR-10 dataset
            train_dataset = datasets.CIFAR10(
                root="./data", train=True, download=True, transform=transform
            )
            test_dataset = datasets.CIFAR10(
                root="./data", train=False, download=True, transform=transform
            )
            num_classes = 10

        elif dataset == "caltech256":
            # Define transforms for the dataset
            # 30607, resize to 32x32
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((32, 32)),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            # Load the Caltech 256 dataset
            d = datasets.Caltech256(
                root="./data", download=True, transform=transform
            )
            train_dataset, test_dataset = torch.utils.data.random_split(d, 
                [int(0.8 * len(d)), len(d)-int(0.8 * len(d))])
            train_dataset = SubsetDataset(train_dataset)
            test_dataset = SubsetDataset(test_dataset)
            num_classes = 256

        elif dataset == "cifar100":
            # Define transforms for the dataset
            # 50K, 10K, already 32x32
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            # Load the CIFAR-100 dataset
            train_dataset = datasets.CIFAR100(
                root="./data", train=True, download=True, transform=transform
            )
            test_dataset = datasets.CIFAR100(
                root="./data", train=False, download=True, transform=transform
            )
            num_classes = 100

        elif dataset == "fashionmnist":
            # Define transforms for the dataset
            # 60K, 10K, already 28x28
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            # Load the FashionMNIST dataset
            train_dataset = datasets.FashionMNIST(
                root="./data", train=True, download=True, transform=transform
            )
            test_dataset = datasets.FashionMNIST(
                root="./data", train=False, download=True, transform=transform
            )
            num_classes = 10

        elif dataset == "imagenet":
            # Define transforms for the dataset
            # 1281167, 50K, 100K, resize to 224x224
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Resize((224, 224)),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            # Load the ImageNet dataset
            train_dataset = datasets.ImageNet(
                root="/scratch/ssd004/datasets/imagenet256", split="train", transform=transform
            )
            test_dataset = datasets.ImageNet(
                root="/scratch/ssd004/datasets/imagenet256", split="val", transform=transform
            )
            num_classes = 1000

        # Source: https://github.com/mlmed/torchxrayvision

        elif dataset == "nih":
            # Define transforms for the dataset
            # 112120, resize to 224x224
            transform = transforms.Compose(
                [
                    xrv.datasets.XRayResizer(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            # National Institutes of Health ChestX-ray8 dataset. https://arxiv.org/abs/1705.02315
            train_dataset = xrv.datasets.NIH_Dataset(
                imgpath="/datasets/nih-chest-xrays", csvpath="/datasets/nih-chest-xrays/train_val_list.txt", transform=transform, unique_patients=False
            )
            test_dataset = xrv.datasets.NIH_Dataset(
                imgpath="/datasets/nih-chest-xrays", csvpath="/datasets/nih-chest-xrays/test.txt", transform=transform, unique_patients=False
            )
            num_classes = 14

        elif dataset == "pneumonia":
            # Define transforms for the dataset
            # 112120, resize to 224x224
            transform = transforms.Compose(
                [
                    xrv.datasets.XRayResizer(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            # RSNA Pneumonia Detection Challenge. https://pubs.rsna.org/doi/full/10.1148/ryai.2019180041
            train_dataset = xrv.datasets.RSNA_Pneumonia_Dataset(
                imgpath="./data/pneumonia/stage_2_train_images_jpg", transform=transform
            )
            test_dataset = xrv.datasets.RSNA_Pneumonia_Dataset(
                imgpath="./data/pneumonia/stage_2_test_images_jpg", transform=transform
            )
            num_classes = 10

        elif dataset == "padchest":
            # Define transforms for the dataset
            # 160K, already 224x224
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            # PadChest: A large chest x-ray image dataset with multi-label annotated reports. https://arxiv.org/abs/1901.07441
            train_dataset = xrv.datasets.PC_Dataset(
                imgpath="./data/padchest/pc/images-224", transform=transform
            )
            test_dataset = xrv.datasets.PC_Dataset(
                imgpath="./data/padchest/pc/images-224", transform=transform
            )
            num_classes = 10

        elif dataset == "vindrcxr":
            # Define transforms for the dataset
            # 160K, already 224x224
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            # VinDr-CXR: An open dataset of chest X-rays with radiologist's annotations. https://arxiv.org/abs/2012.15029
            train_dataset = xrv.datasets.VinBrain_Dataset(
                imgpath=".data/vindrcxr/train", csvpath=".data/vindrcxr/train.csv",  transform=transform
            )
            test_dataset = xrv.datasets.VinBrain_Dataset(
                imgpath=".data/vindrcxr/test", csvpath=".data/vindrcxr/sample_submission.csv", transform=transform
            )
            num_classes = 10

        elif dataset == "objectcxr":
            # Define transforms for the dataset
            # 10K, 
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            # Object-CXR Dataset: Automatic detection of foreign objects on chest X-rays. https://academictorrents.com/details/fdc91f11d7010f7259a05403fc9d00079a09f5d5
            train_dataset = xrv.datasets.ObjectCXR_Dataset(
                imgzippath="./data/objectcxr/train.zip", csvpath="./data/objectcxr/train.csv", transform=transform
            )
            test_dataset = xrv.datasets.ObjectCXR_Dataset(
                imgzippath="./data/objectcxr/dev.zip", csvpath="./data/objectcxr/dev.csv", transform=transform
            )
            num_classes = 10

        elif dataset == "siim":
            # Define transforms for the dataset
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            # SIIM Pneumothorax Dataset. https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation
            train_dataset = xrv.datasets.SIIM_Pneumothorax_Dataset(
                imgpath="./data/siim/stage_2_images", csvpath="./data/siim/stage_2_train.csv", transform=transform
            )
            test_dataset = xrv.datasets.SIIM_Pneumothorax_Dataset(
                imgpath="./data/siim/stage_2_images", csvpath="./data/siim/stage_2_sample_submission.csv", transform=transform
            )
            num_classes = 10

        else:
            raise ValueError("Invalid dataset!")

        total_samples = len(train_dataset)
        print(total_samples)

        # Set device to use
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ####################
        #
        # DATALOADER MODULE
        #
        ####################
        #metadata = {
        #    "p": p,
        #    "hardness": hardness,
        #    "dataset": dataset,
        #    "model": model_name,
        #    "run": i,
        #    "seed": seed,
        #}

        #wandb.log(metadata)

        # Allows importing data in multiple formats
        dataloader_class = MultiFormatDataLoader(
            data=train_dataset,
            target_column=None,
            data_type="torch_dataset",
            data_modality="image",
            dataset_name=dataset,
            batch_size=64,
            shuffle=True,
            num_workers=0,
            transform=None,
            image_transform=None,
            perturbation_method=hardness,
            p=p,
            rule_matrix=rule_matrix,
        )

        dataloader, dataloader_unshuffled = dataloader_class.get_dataloader()
        flag_ids = dataloader_class.get_flag_ids()

        ####################
        #
        # TRAINER MODULE
        #
        ####################

        # Instantiate the neural network
        if dataset == "cifar10" or dataset == "caltech256" or dataset == "cifar100":
            if model_name == "LeNet":
                model = LeNet(num_classes=num_classes).to(device)
            if model_name == "ResNet":
                model = ResNet18(num_classes=num_classes).to(device)
        elif dataset == "mnist" or dataset == "fashionmnist":
            if model_name == "LeNet":
                model = LeNetMNIST(num_classes=num_classes).to(device)
            if model_name == "ResNet":
                model = ResNet18MNIST(num_classes=num_classes).to(device)
        elif dataset == "imagenet":
            if model_name == "LeNet":
                model = LeNetImageNet(num_classes=num_classes).to(device)
            if model_name == "ResNet":
                model = models.resnet18().to(device)
        elif dataset == "xray":
            if model_name == "LeNet":
                model = LeNetMNIST(num_classes=num_classes).to(device)
            if model_name == "ResNet":
                model = ResNet18MNIST(num_classes=num_classes).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Instantiate the PyTorchTrainer class
        trainer = PyTorchTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            lr=0.001,
            epochs=epochs,
            total_samples=total_samples,
            num_classes=num_classes,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        # Train the model
        trainer.fit(dataloader, dataloader_unshuffled, wandb_num=wandb.run.id)

        hardness_dict = trainer.get_hardness_methods()

        ####################
        #
        # EVALUATOR MODULE
        #
        ####################

        eval = Evaluator(hardness_dict=hardness_dict, flag_ids=flag_ids, p=p)

        eval_dict, raw_scores_dict = eval.compute_results()
        # add sleep in case of machine latency
        time.sleep(10)
        print(eval_dict)
        wandb.log(eval_dict)

        scores_dict = {
            "metadata": metadata,
            "scores": raw_scores_dict,
            "flag_ids": flag_ids,
        }
        # add sleep in case of machine latency
        time.sleep(30)
        metainfo = f"{dataset}_{hardness}_{p}_{seed}_{i}"
        # log overall_result_dicts to wandb as a pickle
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            pickle.dump(scores_dict, temp_file)
            temp_file_path = temp_file.name

        # Log the pickle as a wandb artifact
        artifact = wandb.Artifact(f"scores_dict_{metainfo}", type="pickle")
        artifact.add_file(temp_file_path, name=f"scores_dict_{metainfo}.pkl")
        wandb.run.log_artifact(artifact)
        # Clean up the temporary file
        os.remove(temp_file_path)

        # add sleep in case of machine latency
        time.sleep(30)

        wandb.finish()

        if not args.fix_seed:
            seed += 1


def str2bool(v):
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your program.")
    # Add command-line arguments
    parser.add_argument("--total_runs", type=int, default=3, help="Total runs")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--prop", type=float, default=0.1, help="prop")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs")
    parser.add_argument("--hardness", type=str, default="uniform", help="hardness type")
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10", "caltech256", "cifar100", "fashionmnist", "imagenet", "nih", "nihpneumonia", "padchest", "vindrcxr", "objectcxr", "siim"],
        help="Dataset",
    )
    parser.add_argument("--model_name", type=str, default="LeNet", help="Model name")
    parser.add_argument(
        "--fix_seed",
        type=str2bool,
        default="false",
        help="fix the seed for consistency exps",
    )

    args = parser.parse_args()

    main(args)
