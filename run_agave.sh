#!/bin/bash

#SBATCH -N 1            # number of nodes
#SBATCH -c 50  # number of "tasks" (default: allocates 1 core per task)
#SBATCH -t 0-06:00:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition
#SBATCH -q public       # QOS
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user=mgrohols@asu.edu # Mail-to address
#SBATCH --export=NONE   # Purge the job-submitting shell environment

source "$HOME/.cargo/env"
sh ./run_job.sh 0 0