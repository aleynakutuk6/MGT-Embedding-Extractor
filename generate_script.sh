#!/bin/bash
# 
# CompecTA (c) 2018
# 
# You should only work under the /scratch/users/<username> directory.
#
# Jupyter job submission script
#
# TODO:
#   - Set name of the job below changing "JupiterNotebook" value. - Set the requested number of nodes (servers) with --nodes parameter. 
#   - Set the requested number of tasks (cpu cores) with --ntasks parameter. (Total accross all nodes) - Select the partition (queue) 
#   you want to run the job in:
#     - short : For jobs that have maximum run time of 120 mins. Has higher priority. - mid : For jobs that have maximum run time of 1 
#   day.. - long : For jobs that have maximum run time of 7 days. Lower priority than short. - longer: For testing purposes, queue has 
#   31 days limit but only 3 nodes. - Set the required time limit for the job with --time parameter.
#     - Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and 
#   "days-hours:minutes:seconds" - Put this script and all the input file under the same directory. - Set the required parameters, 
#   input/output file names below. - If you do not want mail please remove the line that has --mail-type and --mail-user. If you do 
#   want to get notification emails, set your email address. - Put this script and all the input file under the same directory. - 
#   Submit this file using:
#      sbatch jupyter_submit.sh
#
# -= Resources =-
#
#SBATCH --job-name=mgt-pr
#SBATCH --nodes 1 
#SBATCH --mem=20G 
#SBATCH --ntasks-per-node=4
#SBATCH --partition ai 
#SBATCH --account=ai 
#SBATCH --qos=ai 
#SBATCH --gres=gpu:1
#SBATCH --constraint=tesla_t4
#SBATCH --time=120:00:00 
#SBATCH --output=mgt-processing.log 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=akutuk21@ku.edu.tr

# Please read before you run: http://login.kuacc.ku.edu.tr/#h.3qapvarv2g49
################################################################################
##################### !!! DO NOT EDIT BELOW THIS LINE !!! ######################
################################################################################

# Load Anaconda
echo "=======================" 
echo "Loading Anaconda Module..." 
module unload cuda
module unload cudnn
module add cuda/10.2
module add cudnn/7.6.5/cuda-10.2
module load anaconda/5.2.0 
module add nnpack/latest 
module add rclone 

source activate sketchformer-new

# Set stack size to unlimited
echo "Setting stack size to unlimited..." 
ulimit -s unlimited 
ulimit -l unlimited 
ulimit -a 
echo

################################################################################
##################### !!! DO NOT EDIT ABOVE THIS LINE !!! ######################
################################################################################

python3 extract_mgt_embeddings.py
