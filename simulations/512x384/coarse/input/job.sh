#!/bin/bash
#SBATCH --job-name=Eureca_fine
#SBATCH --output=Eureca_fine.out
#SBATCH --error=Eureca_fine.error
#SBATCH --mem=128gb
#SBATCH --time=48:00:00
#SBATCH --qos=np
#SBATCH --ntasks=256
#SBATCH --ntasks-per-node=128
#SBATCH --nodes=2
#SBATCH --mail-type=FAIL
module load prgenv/gnu
module load hpcx-openmpi
module load cmake/3.19.5
module load netcdf4/4.7.4
module load fftw/3.3.9
mpirun -np 256 ./dales4
./merge.sh
