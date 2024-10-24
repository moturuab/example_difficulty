import argparse
import io
import os
import pickle
import tempfile
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import yaml
from torchvision import datasets, models
import torchxrayvision as xrv
import torchvision.transforms.v2 as transforms
from sklearn.model_selection import train_test_split
from collections import Counter

from src.dataloader import MultiFormatDataLoader, SubsetDataset
from src.dataloader_xray import XrayMultiFormatDataLoader, XraySubsetDataset
from src.evaluator import Evaluator
from src.models import *
from src.trainer import PyTorchTrainer
from src.utils import seed_everything
from src.losses import WeightedCrossEntropyLoss, WeightedFocalLoss

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# declare -a arr=("lenet" "resnet")
# declare -a arr2=("adjacent" "asymmetric" "crop" "idcov" "instance" "oodcov" "uniform" "zoom")
# for i in $(seq 0.1 0.1 0.5); do for j in "${arr[@]}"; do for k in "${arr2[@]}"; do sbatch run_small_"$k"_"$j".sh "$i"; done; done; done

def train_test_dataset(dataset, val_split=0.15, test_split=0.20):
    train_idx, test_idx = train_test_split(list(range(len(dataset))), test_size=test_split)
    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    train_idx, val_idx = train_test_split(list(range(len(train_dataset))), test_size=val_split)

    train_dataset = torch.utils.data.Subset(train_dataset, train_idx)
    val_dataset = torch.utils.data.Subset(train_dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)

    return train_dataset, val_dataset, test_dataset, train_idx, val_idx, test_idx

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
    loss = args.loss
    p = args.prop
    init_alpha = args.init_alpha
    init_beta = args.init_beta
    init_delta = args.init_delta
    lr = args.lr
    wd = args.wd
    alpha_lr = args.alpha_lr
    beta_lr = args.beta_lr
    delta_lr = args.delta_lr
    alpha_wd = args.alpha_wd
    beta_wd = args.beta_wd
    delta_wd = args.delta_wd
    warmup = args.warmup
    focal_gamma = args.focal_gamma
    reweight = args.reweight
    clean_val = args.clean_val
    calibrate = args.calibrate
    groupid = args.groupid
    metainfo = f"{hardness}_{dataset}_{model_name}_{loss}_{p}_{reweight}_{clean_val}_{calibrate}_{init_alpha}_{init_beta}_{init_delta}_{alpha_lr}_{beta_lr}_{delta_lr}_{alpha_wd}_{beta_wd}_{delta_wd}_{lr}_{wd}_{focal_gamma}_{epochs}_{total_runs}_{seed}_{groupid}"

    #os.mkdir(metainfo)

    full_dataset = None
    train_idx = None
    val_idx = None
    test_idx = None

    # new wandb run
    config_dict = {'total_runs': total_runs, 'hardness': hardness, 'loss': loss, 'dataset': dataset, 'reweight': reweight, 
    'calibrate':calibrate, 'init_alpha': init_alpha, 'alpha_lr': alpha_lr, 'beta_lr': beta_lr, 'delta_lr': delta_lr, 'lr': lr, 
    'wd': wd, 'alpha_wd': alpha_wd, 'beta_wd': beta_wd, 'delta_wd': delta_wd, 'fix_seed': args.fix_seed, 'init_beta': init_beta,
    'init_delta': init_delta, 'warmup': warmup, 'focal_gamma': focal_gamma, 'clean_val': clean_val, 'model_name': model_name,
    'total_epochs': epochs, 'seed': seed, 'prop': p, 'groupid': groupid}

    run = wandb.init(
        project="example_difficulty",
        name=metainfo,
        entity=wandb_entity,
        config=config_dict
    )

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
        print(metainfo)
        dir_to_delete = None

        wandb.log({'run': i}, commit=False)
        
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
                rule_matrix = {0: [0], 1: [1], 2: [198], 3: [3], 4: [4], 5: [5], 6: [8, 63, 163], 7: [12, 65, 198, 211], 8: [6, 55], 9: [9], 10: [10], 11: [66], 12: [7, 65], 13: [13], 14: [14], 15: [15], 16: [16], 17: [17], 18: [18], 19: [19], 20: [20], 21: [21], 22: [22], 23: [23], 24: [24], 25: [54], 26: [150, 178], 27: [83, 133], 28: [28], 29: [121], 30: [30], 31: [31], 32: [32], 33: [197, 229], 34: [34], 35: [35], 36: [36], 37: [89], 38: [198], 39: [39], 40: [40], 41: [134, 198], 42: [42], 43: [43], 44: [44], 45: [45], 46: [46], 47: [188], 48: [48], 49: [49], 50: [50], 51: [51], 52: [52], 53: [53], 54: [25], 55: [8, 253], 56: [56], 57: [57], 58: [58], 59: [88], 60: [60], 61: [61], 62: [62], 63: [6], 64: [83], 65: [7, 12, 198, 211], 66: [11], 67: [67], 68: [68], 69: [69], 70: [70], 71: [71], 72: [72], 73: [130], 74: [74], 75: [75], 76: [76], 77: [77], 78: [78], 79: [255], 80: [80], 81: [158], 82: [82], 83: [27, 64, 84, 133], 84: [83], 85: [85], 86: [86], 87: [87], 88: [59], 89: [37], 90: [90], 91: [91], 92: [92], 93: [93], 94: [94], 95: [95], 96: [96], 97: [135], 98: [98], 99: [99], 100: [100], 101: [250], 102: [102], 103: [103], 104: [249], 105: [105], 106: [106], 107: [107], 108: [108], 109: [239], 110: [110], 111: [111], 112: [112], 113: [206], 114: [114], 115: [115], 116: [116], 117: [117], 118: [118], 119: [119], 120: [120], 121: [29], 122: [122], 123: [123], 124: [124], 125: [125], 126: [126], 127: [127], 128: [167], 129: [129], 130: [73], 131: [131], 132: [132], 133: [27, 83], 134: [41, 198], 135: [97], 136: [176], 137: [137], 138: [138], 139: [139], 140: [140], 141: [141], 142: [142], 143: [143], 144: [144], 145: [145], 146: [146], 147: [149], 148: [148], 149: [147], 150: [26, 178, 204], 151: [151], 152: [152], 153: [153], 154: [154], 155: [155], 156: [156], 157: [157], 158: [81], 159: [159], 160: [160], 161: [161], 162: [162], 163: [6], 164: [184, 198, 228, 243], 165: [165], 166: [166], 167: [128], 168: [168], 169: [169], 170: [170], 171: [172], 172: [171], 173: [173], 174: [174], 175: [202], 176: [136], 177: [177], 178: [26, 150, 204], 179: [233], 180: [180], 181: [181], 182: [182], 183: [183], 184: [164, 198, 228, 243], 185: [185], 186: [186], 187: [187], 188: [47], 189: [189], 190: [190], 191: [191], 192: [192], 193: [193], 194: [194], 195: [205], 196: [196], 197: [33, 229], 198: [2, 7, 38, 41, 65, 134, 164, 184, 228], 199: [199], 200: [200], 201: [201], 202: [175, 230], 203: [203], 204: [150, 178], 205: [195], 206: [113], 207: [207], 208: [208], 209: [209], 210: [246], 211: [7, 65], 212: [212], 213: [213], 214: [214], 215: [215], 216: [216], 217: [217], 218: [218], 219: [219], 220: [220], 221: [221], 222: [222], 223: [223], 224: [224], 225: [225], 226: [226], 227: [227], 228: [164, 184, 198, 243], 229: [33, 197], 230: [202], 231: [231], 232: [232], 233: [179], 234: [234], 235: [235], 236: [236], 237: [237], 238: [238], 239: [109], 240: [240], 241: [241], 242: [242], 243: [164, 184, 228], 244: [244], 245: [245], 246: [210], 247: [247], 248: [248], 249: [104], 250: [101], 251: [251], 252: [252], 253: [55], 254: [254], 255: [79], 256: [256]}
            
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

                rule_matrix = {i:[] for i in range(100)}
                for i in temp_matrix.keys():
                    for j in range(5):
                        rule_matrix[temp_matrix[i][j]] = temp_matrix[i].copy()
                        rule_matrix[temp_matrix[i][j]].remove(temp_matrix[i][j])


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

            # (1, Atelectasis; 2, Cardiomegaly; 3, Effusion; 4, Infiltration; 5, Mass; 6, Nodule; 7, Pneumonia; 8, Pneumothorax; 9, Consolidation; 10, Edema; 11, Emphysema; 12, Fibrosis; 13, Pleural_Thickening; 14 Hernia)
            if dataset == "nih":
                rule_matrix = {
                0: [2,7],
                1: [1],
                2: [0,7],
                3: [6,8,9,10,11],
                4: [5],
                5: [4],
                6: [3,8,9,10,11],
                7: [0,2],
                8: [3,6,9,10,11],
                9: [3,6,8,10,11],
                10: [3,6,8,9,11],
                11: [3,6,8,9,10],
                12: [12],
                13: [13]
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
            full_dataset = d

            train_dataset, val_dataset, test_dataset, train_idx, val_idx, test_idx = train_test_dataset(d)
            num_classes = 257

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
                    transforms.Normalize((0.5,), (0.5,)),
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
                    #xrv.datasets.XRayResizer(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            # National Institutes of Health ChestX-ray8 dataset. https://arxiv.org/abs/1705.02315
            train_dataset = xrv.datasets.NIH_Dataset(
                imgpath="/datasets/NIH/images-224", transform=None, unique_patients=True
            )
            test_dataset = xrv.datasets.NIH_Dataset(
                imgpath="/datasets/NIH/images-224", transform=None, unique_patients=True
            )
            num_classes = 18

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
            num_classes = 2

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
            # 15K, 3K, already 224x224
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
            # VinDr-CXR: An open dataset of chest X-rays with radiologist's annotations. https://arxiv.org/abs/2012.15029
            train_dataset = xrv.datasets.VinBrain_Dataset(
                imgpath=".data/vindrcxr/train", csvpath=".data/vindrcxr/train.csv", transform=transform
            )
            test_dataset = xrv.datasets.VinBrain_Dataset(
                imgpath=".data/vindrcxr/test", csvpath=".data/vindrcxr/sample_submission.csv", transform=transform
            )
            num_classes = 28

        elif dataset == "objectcxr":
            # Define transforms for the dataset
            # 10K, resize to 224x224
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
            num_classes = 2

        elif dataset == "siim":
            # Define transforms for the dataset
            # 3205, resize to 224x224
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
            num_classes = 2

        else:
            raise ValueError("Invalid dataset!")

        total_samples = len(train_dataset)
        n = total_samples
        print(total_samples)

        # Set device to use
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ####################
        #
        # DATALOADER MODULE
        #
        ####################
        
        metadata = {
            "p": p,
            "hardness": hardness,
            "dataset": dataset,
            "model": model_name,
            "run": i,
            "seed": seed,
        }

        #wandb.log(metadata)

        if dataset == "caltech256":
            l = np.array(range(len(full_dataset)))
            np.random.shuffle(l)
            temp_train_idx = np.array(l[:int(0.8*0.85*n)])
            temp_val_idx = np.array(l[int(0.8*0.85*n):int(0.8*n)])
            temp_test_idx = np.array(l[int(0.8*n):])

            temp_train_dataset = SubsetDataset(full_dataset, temp_train_idx, torch.from_numpy(np.array([full_dataset[i][1] for i in temp_train_idx])))
            temp_val_dataset = SubsetDataset(full_dataset, temp_val_idx, torch.from_numpy(np.array([full_dataset[i][1] for i in temp_val_idx])))
            temp_test_dataset = SubsetDataset(full_dataset, temp_test_idx, torch.from_numpy(np.array([full_dataset[i][1] for i in temp_test_idx])))
        elif dataset == "nih":
            labels = torch.from_numpy(np.array(train_dataset.labels))
            row_sums = torch.sum(labels, dim=1)
            indices = torch.nonzero(row_sums <= 1).squeeze()
            l = np.array(indices)
            n = len(indices)
            print(n)
            np.random.shuffle(l)

            temp_train_idx = np.array(l[:int(0.8*0.85*n)])
            temp_val_idx = np.array(l[int(0.8*0.85*n):int(0.8*n)])
            temp_test_idx = np.array(l[int(0.8*n):])

            temp_train_dataset = XraySubsetDataset(train_dataset, temp_train_idx, torch.from_numpy(np.array(train_dataset.labels))[temp_train_idx])
            temp_val_dataset = XraySubsetDataset(train_dataset, temp_val_idx, torch.from_numpy(np.array(train_dataset.labels))[temp_val_idx])
            temp_test_dataset = XraySubsetDataset(train_dataset, temp_test_idx, torch.from_numpy(np.array(train_dataset.labels))[temp_test_idx])
        else:
            l = np.array(range(n))
            np.random.shuffle(l)
            temp_train_idx = np.array(l[:int(0.85*n)])
            temp_val_idx = np.array(l[int(0.85*n):])
            temp_test_idx = np.array(range(len(test_dataset)))

            temp_train_dataset = SubsetDataset(train_dataset, temp_train_idx, torch.from_numpy(np.array(train_dataset.targets))[temp_train_idx])
            temp_val_dataset = SubsetDataset(train_dataset, temp_val_idx, torch.from_numpy(np.array(train_dataset.targets))[temp_val_idx])
            temp_test_dataset = SubsetDataset(test_dataset, temp_test_idx, torch.from_numpy(np.array(test_dataset.targets))[temp_test_idx])
        
        if dataset == "nih":
            dataloader_class = XrayMultiFormatDataLoader(
            data=temp_train_dataset,
            full_dataset=full_dataset,
            idx=temp_train_idx,
            target_column=None,
            data_type="torch_dataset",
            data_modality="image",
            dataset_name=dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            transform=None,
            image_transform=None,
            perturbation_method=None,
            p=None,
            rule_matrix=None)

            val_dataloader_class = XrayMultiFormatDataLoader(
            data=temp_val_dataset,
            full_dataset=full_dataset,
            idx=temp_val_idx,
            target_column=None,
            data_type="torch_dataset",
            data_modality="image",
            dataset_name=dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            transform=None,
            image_transform=None,
            perturbation_method=None,
            p=None,
            rule_matrix=None)

            test_dataloader_class = XrayMultiFormatDataLoader(
            data=temp_test_dataset,
            full_dataset=full_dataset,
            idx=temp_test_idx,
            target_column=None,
            data_type="torch_dataset",
            data_modality="image",
            dataset_name=dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            transform=None,
            image_transform=None,
            perturbation_method=None,
            p=None,
            rule_matrix=None)

        else:
            dataloader_class = MultiFormatDataLoader(
            data=temp_train_dataset,
            full_dataset=full_dataset,
            idx=temp_train_idx,
            target_column=None,
            data_type="torch_dataset",
            data_modality="image",
            dataset_name=dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            transform=None,
            image_transform=None,
            perturbation_method=hardness,
            p=p,
            rule_matrix=rule_matrix)

            val_dataloader_class = MultiFormatDataLoader(
            data=temp_val_dataset,
            full_dataset=full_dataset,
            idx=temp_val_idx,
            target_column=None,
            data_type="torch_dataset",
            data_modality="image",
            dataset_name=dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            transform=None,
            image_transform=None,
            perturbation_method=hardness,
            p=p,
            rule_matrix=rule_matrix)

            test_dataloader_class = MultiFormatDataLoader(
            data=temp_test_dataset,
            full_dataset=full_dataset,
            idx=temp_test_idx,
            target_column=None,
            data_type="torch_dataset",
            data_modality="image",
            dataset_name=dataset,
            batch_size=64,
            shuffle=False,
            num_workers=0,
            transform=None,
            image_transform=None,
            perturbation_method=hardness,
            p=p,
            rule_matrix=rule_matrix)

        dataloader, dataloader_unshuffled = dataloader_class.get_dataloader()
        train_flag_ids = dataloader_class.get_flag_ids()
        val_dataloader, val_dataloader_unshuffled = val_dataloader_class.get_dataloader()
        val_flag_ids = val_dataloader_class.get_flag_ids()
        test_dataloader, test_dataloader_unshuffled = test_dataloader_class.get_dataloader()
        test_flag_ids = test_dataloader_class.get_flag_ids()

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
        elif dataset == "nih":
            model = xrv.models.DenseNet().to(device)

        alpha = nn.Parameter(torch.tensor(init_alpha), requires_grad=True)
        beta = nn.Parameter(torch.tensor(init_beta), requires_grad=True)
        delta = nn.Parameter(torch.tensor(init_delta), requires_grad=True)
        if loss == 'CE':
            criterion = WeightedCrossEntropyLoss(reweight=reweight, alpha=alpha, beta=beta, delta=delta, num_classes=num_classes, warmup=warmup, device=device)
        elif loss == 'FL':
            criterion = WeightedFocalLoss(reweight=reweight, alpha=alpha, beta=beta, delta=delta, gamma=focal_gamma, num_classes=num_classes, warmup=warmup, device=device)

        if reweight:
            optimizer = optim.Adam(list(model.parameters()) + list([alpha, beta, delta]), lr=lr, weight_decay=wd)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

        # Instantiate the PyTorchTrainer class
        trainer = PyTorchTrainer(
            model=model,
            alpha=alpha,
            beta=beta,
            delta=delta,
            flag_ids=train_flag_ids,
            criterion=criterion,
            optimizer=optimizer,
            lr=lr,
            alpha_lr=alpha_lr,
            beta_lr=beta_lr,
            delta_lr=delta_lr,
            wd=wd,
            alpha_wd=alpha_wd,
            beta_wd=beta_wd,
            delta_wd=delta_wd,
            warmup=warmup, 
            epochs=epochs,
            total_samples=len(temp_train_idx),
            num_classes=num_classes,
            reweight=reweight,
            clean_val=clean_val,
            calibrate=calibrate,
            device=device,
            metainfo=metainfo,
        )

        # Train the model
        trainer.fit(dataloader, dataloader_unshuffled, val_dataloader, val_dataloader_unshuffled, test_dataloader, test_dataloader_unshuffled, wandb_num=[wandb.run.id,i])

        hardness_dict = trainer.get_hardness_methods()

        ####################
        #
        # EVALUATOR MODULE
        #
        ####################

        eval = Evaluator(hardness_dict=hardness_dict, flag_ids=train_flag_ids, p=p)

        eval_dict, raw_scores_dict = eval.compute_results()
        # add sleep in case of machine latency
        time.sleep(10)
        
        wandb.log(eval_dict)

        scores_dict = {
            "metadata": metadata,
            "scores": raw_scores_dict,
            "flag_ids": train_flag_ids,
        }
        # add sleep in case of machine latency
        time.sleep(30)

        # log overall_result_dicts to wandb as a pickle
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            pickle.dump(scores_dict, temp_file)
            temp_file_path = temp_file.name

        # Log the pickle as a wandb artifact
        artifact = wandb.Artifact(f"scores_dict_{metainfo.replace(':', '').replace('.', '')}", type="pickle")
        artifact.add_file(temp_file_path, name=f"scores_dict_{metainfo.replace('.', '').replace(':', '')}.pkl")
        wandb.run.log_artifact(artifact)
        # Clean up the temporary file
        os.remove(temp_file_path)

        # add sleep in case of machine latency
        time.sleep(30)

        if not args.fix_seed:
            seed += 1

    wandb.finish()


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
    parser.add_argument("--reweight", action='store_true', help="reweight")
    parser.add_argument(
        "--loss",
        type=str,
        default="CE",
        choices=["MSE", "CE", "FL"],
        help="type of loss function to use",
    )
    parser.add_argument("--clean_val", action='store_true', help="optimize on clean validation set")
    parser.add_argument("--calibrate", action='store_true', help="calibrate on validation set")
    parser.add_argument("--init_alpha", type=float, default=2.0, help="initialize alpha")
    parser.add_argument("--init_beta", type=float, default=2.0, help="initialize beta")
    parser.add_argument("--init_delta", type=float, default=2.0, help="initialize delta")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for network")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay for network")
    parser.add_argument("--alpha_lr", type=float, default=0.01, help="learning rate for alpha")
    parser.add_argument("--beta_lr", type=float, default=0.01, help="learning rate for beta")
    parser.add_argument("--delta_lr", type=float, default=0.01, help="learning rate for delta")
    parser.add_argument("--alpha_wd", type=float, default=0.01, help="weight decay for alpha")
    parser.add_argument("--beta_wd", type=float, default=0.01, help="weight decay for beta")
    parser.add_argument("--delta_wd", type=float, default=0.01, help="weight decay for delta")
    parser.add_argument("--warmup", type=int, default=0, help="weighting starts after warmup epochs")
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="gamma for focal loss")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs")
    parser.add_argument("--hardness", type=str, default="uniform", help="hardness type")
    parser.add_argument("--groupid", type=str, default="0", help="group id (time)")
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
