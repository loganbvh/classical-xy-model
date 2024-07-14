#!/bin/bash
#SBATCH --job-name=xy_loop_array
#SBATCH --array=0-200
#SBATCH --output=slurm-%A_%a.out
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6GB
#SBATCH --time=24:00:00

# Activate conda env
source $GROUP_HOME/miniconda3/bin/activate
conda activate superscreen

outdir=$GROUP_SCRATCH/xy/results/$SLURM_ARRAY_JOB_ID
mkdir -p $outdir

pyscript=$HOME/classical-xy-model/xy_loop.py

# Copy the python script and this shell script to the results directory
cp -u $pyscript $outdir/
cp -u $0 $outdir/

python $pyscript \
    --output-path=$outdir/results-$SLURM_ARRAY_TASK_ID.json \
    --temperature=$(python -c "import numpy; print(numpy.linspace(0.01, 2.01, 201))[$SLURM_ARRAY_TASK_ID]") \
    --nrows=50 \
    --ncols=100 \
    --loop-center 0 0 2 \
    --loop-radius=7.5 \
    --loop-current=100 \
    --pickup-loop-center 0 0 2 \
    --pickup-loop-radius=3.5 \
    --thermalize-passes=100000 \
    --measure-passes=1000000 \
    --hot-start \
    --periodic \
    --progress-bar

# Move the stdout log to the results directory
mv "slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" $outdir