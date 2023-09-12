#!/bin/bash
#SBATCH -J torchrun-pcqm4mv2_graphormer_base
#SBATCH --time=5:00:00
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-cpu=36000
#S BATCH --exclusive=user
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
# module load Anaconda3/2022.10
# conda activate /data/stat-cadd/shil5919/molfeat
# pip3 install torch torchvision torchaudio
# pip install molfeat[graphormer]
# pip install numpy
# echo $CONDA_DEFAULT_ENV
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# $(scontrol show job $SLURM_JOBID | awk -F= '/BatchHost/ {print $2}')
# $(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADD=htc-g055.arcus-htc.arc.local
echo "MASTER_ADD="$MASTER_ADD
echo "training"
# python -m torch.distributed.launch 
torchrun train.py --batch-size 96 --data-path data/chembl_02 --save-directory pcqm4mv2_graphormer_base --model-choice transformer --num-epoch 30 transformer