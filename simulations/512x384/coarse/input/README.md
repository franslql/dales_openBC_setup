# input

## backrad.inp.000.nc
Input profile for radiation routine.

## baseprof.inp.000
Reference density profile (not required as input).

## exnr.inp.000
Exner function used for calculation DALES input (not required as input).

## job.sh
Job script to run simulation on ATOS (ECMWF).

## lscale.inp.000
Large scale forcing profiles.

## merge.sh
Merges the seperate DALES output files.

## namoptions
Settings for the simulation.

## prof.inp.000
Initialisation profiles, used to calculate baseprof.inp.001 and fields not present in initfields.inp.001.nc

## rrtmg_lw.nc
Required for longwave radiation rrtmg routine.

## rrtmg_sw.nc
Required for shortwave radiation rrtmg routine.

## scalar.inp.000
Initialisation profiles for scalars. Used for scalar fields not present in initfields.inp.001.nc.

## openboundaries.inp.000.nc (not present due to size)
Boundary conditions for DALES. Created with scripts/createBoundaryInput.py.

## initfields.inp.000.nc (not present due to size)
Initialisation fields for DALES. Created with scripts/createBoundaryInput.py.
