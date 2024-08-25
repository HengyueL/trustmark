#!/bin/bash -l
#SBATCH --time=24:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8

#SBATCH --mem=16g
#SBATCH --tmp=16g
#SBATCH --mail-type=ALL  
#SBATCH --mail-user=liang656@umn.edu 

#SBATCH -p agsmall

#SBATCH --job-name="prep_trustmark"
#SBATCH --output=out_prep_trustmark.out
#SBATCH --error=err_prep_trustmark.err

cd /scratch.global/liang656/trustmark/python/
export PATH=/home/jusun/liang656/miniconda3/envs/trustmark/bin:$PATH

python msi_experiments/collect_decode_clean.py --dataset DiffusionDB

