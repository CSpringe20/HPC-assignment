#!/bin/bash
#SBATCH --job-name=stencil_scaling
#SBATCH --time=0:20:00
#SBATCH --nodes=4                # Always reserve 4 nodes
#SBATCH --ntasks-per-node=16     # 16 MPI ranks per node (128 ranks total)
#SBATCH --cpus-per-task=8        # 8 OpenMP threads per rank
#SBATCH --mem=0
#SBATCH --partition=EPYC
#SBATCH -A dssc
#SBATCH --exclusive
#SBATCH --output=mpi-%j.out
#SBATCH --error=mpi-%j.err

module load openMPI/5.0.5

mpicc -fopenmp -Iinclude src/stencil_template_parallel.c -o stencil

# OpenMP settings
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_NUM_THREADS=8

RUN=./stencil
BASE_X=30000
BASE_Y=30000
ARGS_COMMON="-e 400 -n 100 -o 0"

STRONG_CSV="csv/strong_scaling.csv"
WEAK_CSV="csv/weak_scaling.csv"

# CSV headers
echo "NODES,MPI_TASKS,OMP_THREADS,GRID_X,GRID_Y,MAX_TOTAL,AVG_TOTAL,MAX_COMP,AVG_COMP,MAX_COMM,AVG_COMM" > $STRONG_CSV
echo "NODES,MPI_TASKS,OMP_THREADS,GRID_X,GRID_Y,MAX_TOTAL,AVG_TOTAL,MAX_COMP,AVG_COMP,MAX_COMM,AVG_COMM" > $WEAK_CSV

# STRONG SCALING: Fixed total size
for N in 1 2 3 4; do
  TOTAL_TASKS=$((N * 16))
  for r in 1 2 3; do
    srun --ntasks=$TOTAL_TASKS --cpus-per-task=8 --mpi=pmix_v5 \
      $RUN -x $BASE_X -y $BASE_Y $ARGS_COMMON >> $STRONG_CSV
  done
done

# WEAK SCALING: Increase X, keep work per node constant
for N in 1 2 3 4; do
  TOTAL_TASKS=$((N * 16))

  X=$((BASE_X * N))
  Y=$BASE_Y

  for r in 1 2 3; do
    srun --ntasks=$TOTAL_TASKS --cpus-per-task=8 --mpi=pmix_v5 \
      $RUN -x $X -y $Y $ARGS_COMMON >> $WEAK_CSV
  done
done



