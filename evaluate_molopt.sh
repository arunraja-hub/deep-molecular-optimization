#!/bin/bash
#SBATCH -J molopt
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-cpu=12000
#S BATCH --exclusive
#SBATCH --chdir=/vols/opig/users/raja/deep-molecular-optimization
#SBATCH --mail-user=arun.raja@dtc.ox.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --partition=high-opig-gpu
#SBATCH --clusters=srf_gpu_01
#SBATCH --partition=high-opig-test
#SBATCH -w nagagpu04.cpu.stats.ox.ac.uk
#SBATCH --output=/vols/opig/users/raja/molopt_slurm/slurm_%j.out                             
#SBATCH --error=/vols/opig/users/raja/molopt_slurm/slurm_%j.err

echo $CUDA_VISIBLE_DEVICES 
echo $CONDA_DEFAULT_ENV
for dir in */; do
  echo "$dir"
done
#python -m venv gin7
#source gin7/bin/activate
source activate molopt
echo $CONDA_DEFAULT_ENV
echo "evaluating"
python evaluate.py --data-path experiments/evaluation_transformer/test/evaluation_60/generated_molecules_prop.csv
python evaluate.py --data-path experiments/evaluation_transformer/test_not_in_train/evaluation_60/generated_molecules_prop.csv
python evaluate.py --data-path experiments/evaluation_transformer/test_unseen_L-1_S01_C10_range/evaluation_60/generated_molecules_prop.csv --range-evaluation lower