#!/bin/bash
#SBATCH -J gpt2-dist
#SBATCH --time=10:00:00
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=1
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
module load Anaconda3/2022.10
conda activate /data/stat-cadd/shil5919/molfeat
pip3 install torch torchvision torchaudio
pip install molfeat[graphormer]
pip install molfeat[transformer]
pip install numpy
echo $CONDA_DEFAULT_ENV
# echo "preprocess"
# python preprocess.py --input-data-path data/chembl_02/mmp_prop.csv
# echo "train"
# python train.py --data-path data/chembl_02 --save-directory pcqm_subset_single_gpu --model-choice transformer transformer
# echo "gen"
# python generate.py --model-choice transformer --data-path data/chembl_02 --test-file-name test --model-path experiments/pcqm_subset_single_gpu/checkpoint --save-directory evaluation_transformer --epoch 10
echo "eval"
python evaluate.py --data-path experiments/evaluation_transformer/test/evaluation_10/generated_molecules_epoch10.csv
