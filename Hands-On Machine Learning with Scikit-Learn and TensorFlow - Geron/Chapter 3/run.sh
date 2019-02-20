#!/bin/bash

# Name of job:
#PBS -N Hands-On_Ex1

# Run for x duration:
#PBS -l walltime=1:00:00

# Select cores
#PBS -l select=1:ncpus=8

# Switch directories
cd $PBS_O_WORKDIR

# Run program
export OMP_NUM_THREADS=8
./Ex1.py

echo Done!