#!/bin/bash

# EXAMPLE USAGE BELOW: 
# run this script from the root directory of the project
# bash run.sh

# Datasets:
# 

# Models:
#

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
groupid=$(date +%F_%T)
fuser -v /dev/nvidia0 -k
python run_experiment.py --groupid $groupid $@
#--total_runs  --hardness  --loss  --dataset  --model_name  --init_alpha  --init_beta  --alpha_lr  --beta_lr  --lr  --focal_gamma  --seed  --prop  --epochs  --reweight --clean_val