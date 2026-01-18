#!/bin/bash
#SBATCH --job-name=stencil_scaling
#SBATCH --time=02:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --partition=EPYC
#SBATCH -A dssc
#SBATCH --exclusive
#SBATCH --output=logs/mpi-%j.out
#SBATCH --error=logs/mpi-%j.err

module load openMPI/5.0.5

mpicc -fopenmp -Iinclude src/stencil_template_parallel.c -o stencil

# OpenMP
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_NUM_THREADS=8

RUN=./stencil
BASE_X=100000
BASE_Y=50000
ARGS_COMMON="-e 4000 -n 100 -o 0"

STRONG_CSV="csv/strong_scaling.csv"
WEAK_CSV="csv/weak_scaling.csv"

# CSV headers
echo "NODES,MPI_TASKS,OMP_THREADS,GRID_X,GRID_Y,MAX_TOTAL,AVG_TOTAL,MAX_COMP,AVG_COMP,MAX_COMM,AVG_COMM" > $STRONG_CSV
echo "NODES,MPI_TASKS,OMP_THREADS,GRID_X,GRID_Y,MAX_TOTAL,AVG_TOTAL,MAX_COMP,AVG_COMP,MAX_COMM,AVG_COMM" > $WEAK_CSV

# STRONG SCALING
for N in 1 2 3 4; do
  TOTAL_TASKS=$((N * 16))
  for r in 1 2 3; do
    srun --ntasks=$TOTAL_TASKS --cpus-per-task=8 --mpi=pmix_v5 \
      $RUN -x $BASE_X -y $BASE_Y $ARGS_COMMON >> $STRONG_CSV
  done
done

# WEAK SCALING
for N in 1 2 3 4; do
  TOTAL_TASKS=$((N * 16))

  X=$((BASE_X * N))
  Y=$BASE_Y

  for r in 1 2 3; do
    srun --ntasks=$TOTAL_TASKS --cpus-per-task=8 --mpi=pmix_v5 \
      $RUN -x $X -y $Y $ARGS_COMMON >> $WEAK_CSV
  done
done



