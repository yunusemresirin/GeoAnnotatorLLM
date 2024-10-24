#!/bin/bash
#SBATCH --job-name=Retrain-Job
#SBATCH --partition=gpu4
#SBATCH --nodelist=wr14
#SBATCH --gres=gpu:1
#SBATCH --mem=25G
#SBATCH --time=01:00:00
#SBATCH --output=slurm.%j.out
#SBATCH --error=slurm.%j.err

# Starting Retrain-Job of LLM
echo "Running finetuning job on node: $SLURM_JOB_NUM_NODES"
# python GeoAnnotator-Finetuning.py $1
python Install-Llama-3-1-8B.py