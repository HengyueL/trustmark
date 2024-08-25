#!/bin/bash -l
#SBATCH --time=24:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

#SBATCH --mem=16g
#SBATCH --tmp=16g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=liang656@umn.edu 

#SBATCH -p apollo_agate
#SBATCH --gres=gpu:a100:1

#SBATCH --job-name="trust-diff"
#SBATCH --output=out_trust-diff.out
#SBATCH --error=err_trust-diff.err

cd /scratch.global/liang656/trustmark/python/
export PATH=/home/jusun/liang656/miniconda3/envs/trustmark/bin:$PATH

python msi_experiments/decode_interm.py --dataset COCO --evade_method dip --arch vanila --start 1900 --end 2000
