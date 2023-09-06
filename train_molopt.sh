#!/bin/bash
#SBATCH -J molopt
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-cpu=36000
#S BATCH --exclusive
#SBATCH --chdir=/home/shil5919/opig/deep-molecular-optimization
#SBATCH --mail-user=arun.raja@dtc.ox.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#S BATCH --partition=high-opig-gpu
#SBATCH --qos=standard
#S BATCH --partition=high-opig-test
#S BATCH -w nagagpu04.cpu.stats.ox.ac.uk
#SBATCH --output=/home/shil5919/opig/slurm_molopt/slurm_%j.out
#SBATCH --error=/home/shil5919/opig/slurm_molopt/slurm_%j.err

echo $CUDA_VISIBLE_DEVICES 
echo $CONDA_DEFAULT_ENV
for dir in */; do
  echo "$dir"
done
#python -m venv gin7
#source gin7/bin/activate
source activate molopt
echo $CONDA_DEFAULT_ENV
echo "training"
pip install optuna
python train.py --data-path data/chembl_02 --save-directory train_transformer_molopt_original --model-choice transformer --num-epoch 30 transformer
