#!/bin/bash

# EXAMPLE USAGE BELOW: 
# run this script from the root directory of the project
# bash run.sh

# Hardness types:
# - "uniform": Uniform mislabeling
# - "asymmetric": Asymmetric mislabeling
# - "adjacent" : Adjacent mislabeling
# - "instance": Instance-specific mislabeling
# - "ood_covariate": Near-OOD Covariate Shift
# - "domain_shift": Specific type of Near-OOD
# - "far_ood": Far-OOD shift (out-of-support)
# - "zoom_shift": Zoom shift  - type of Atypical for images
# - "crop_shift": Crop shift  - type of Atypical for images

# --fix_seed for conistency: where the seed is fixed and we assess model randomness

# Set the parameterizable arguments

#SBATCH --ntasks=1
#SBATCH --mem=64G  
#SBATCH -c 2
#SBATCH --time=4:00:00  
#SBATCH --partition=t4v1,t4v2,rtx6000  
#SBATCH --qos=m3  
#SBATCH --export=ALL  
#SBATCH --output=%x.%j.log  
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL

conda activate py38

total_runs=3
epochs=10
seed=0

# uniform 
hardness="crop_shift"
model_name="LeNet"
fuser -v /dev/nvidia0 -k
groupid=$(date +%F_%T)

dataset="mnist"
python run_experiment.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop $@ --epochs $epochs --groupid $groupid
dataset="cifar10"
python run_experiment.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop $@ --epochs $epochs --groupid $groupid
dataset="caltech256"
python run_experiment.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop $@ --epochs $epochs --groupid $groupid
dataset="cifar100"
python run_experiment.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop $@ --epochs $epochs --groupid $groupid
dataset="fashionmnist"
python run_experiment.py --total_runs $total_runs --hardness $hardness --dataset $dataset --model_name $model_name --seed $seed --prop $@ --epochs $epochs --groupid $groupid
