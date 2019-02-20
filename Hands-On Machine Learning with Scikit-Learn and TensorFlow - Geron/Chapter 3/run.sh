#!/bin/bash

# Name of job:
#PBS -N Hands-On_Ex1

# Run for x duration:
#PBS -l walltime=1:00:00

# Select cores
#PBS -l select=1:ncpus=16:mem=16GB

# Notify me
#PBS -m be
#PBS -M 22115536@sun.ac.za

# Logs and outputs
#PBS -e Ex1.err
#PBS -o Ex1.out

# Switch directories
cd $PBS_O_WORKDIR

# Run program
export OMP_NUM_THREADS=8
./Ex1.py

echo Done!