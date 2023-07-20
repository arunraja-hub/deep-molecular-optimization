#!/bin/bash
#SBATCH -J molopt
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-cpu=1000                                    
#S BATCH --exclusive                     # node should not be shared with other jobs, only use this if you intend the node to be usable only by you as this will block other users from submitting jobs     to the same node                
#SBATCH --chdir=/vols/opig/users/raja/deep-molecular-optimization # From where you want the job to be run
#SBATCH --mail-user=arun.raja@dtc.ox.ac.uk  # set email address                           
#SBATCH --mail-type=ALL                 # Spam us with everything, caution
#SBATCH --mail-type=begin               # Instead only email when job begins...
#SBATCH --mail-type=end                 # ... and ends
#SBATCH --partition=high-opig-gpu  # Select a specific partition rather than default 
#SBATCH --clusters=srf_gpu_01
#SBATCH --partition=high-opig-test    # Select a specific partition rather than default
#SBATCH -w nagagpu04.cpu.stats.ox.ac.uk # Provide a specific node/nodelist rather than the standard nodelist associated with the partition (useful if you have a data setup on one specific node)
#S BATCH --output=/vols/opig/users/raja/slurm_outs/slurm_%j.out  # Writes standard output to this file. %j is jobnumber                             
#S BATCH --error=/vols/opig/users/raja/slurm_outs/slurm_%j.err   # Writes error messages to this file. %j is jobnumber
echo $CUDA_VISIBLE_DEVICES 
echo $CONDA_DEFAULT_ENV
# for dir in */; do
#   echo "$dir"
# done
# #python -m venv gin7
# #source gin7/bin/activate
# source activate molopt
# echo $CONDA_DEFAULT_ENV
# echo "training"
# python train.py --data-path data/chembl_02 --save-directory train_transformer --model-choice transformer --num-epoch 1 transformer
