#!/bin/bash
#SBATCH --job-name=openmp
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --partition=EPYC
#SBATCH -A dssc
#SBATCH --exclusive
#SBATCH --output=logs/openmp-%j.out
#SBATCH --error=logs/openmp-%j.err

module load openMPI/5.0.5

mpicc -fopenmp -Iinclude src/stencil_template_parallel.c -o stencil

# OpenMP
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

RUN=./stencil
ARGS="-x 100000 -y 10000 -e 4000 -n 100 -o 0"
CSV_FILE="csv/openmp_scaling.csv"

echo "NODES,MPI_TASKS,OMP_THREADS,GRID_X,GRID_Y,MAX_TOTAL,AVG_TOTAL,MAX_COMP,AVG_COMP,MAX_COMM,AVG_COMM" > $CSV_FILE

for T in 1 2 4 8 16 32 64 128; do
  export OMP_NUM_THREADS=$T
  for r in 1 2 3; do
    srun --ntasks=1 --cpus-per-task=$T $RUN $ARGS >> $CSV_FILE
  done
done
