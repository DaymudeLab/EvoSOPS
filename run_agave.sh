#!/bin/bash

#SBATCH -N 1  # number of nodes
#SBATCH -c 100
#SBATCH -n 1  # number of "tasks" (default: allocates 1 core per task)
#SBATCH -t 7-00:00:00   # time in d-hh:mm:ss
#SBATCH -p general #highmem 
#SBATCH -q public  #grp_pshakari
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user=%u@asu.edu # Mail-to address
#SBATCH --export=NONE   # Purge the job-submitting shell environment
#SBATCH --mem=50G

# module purge

# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

source "$HOME/.cargo/env"
sh ./run_job.sh