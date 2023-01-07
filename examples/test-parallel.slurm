#!/bin/bash -l
#SBATCH --time=1:00:00
#SBATCH --ntasks=10
#SBATCH --mem=32g
#SBATCH --tmp=50g
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sette095@umn.edu

python parallel_example.py
