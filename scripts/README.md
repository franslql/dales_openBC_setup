# scripts

## createBoundaryInput.py
Creates the input for a DALES simulation with open boundary conditions from HARMONIE output. Used to nest DALES in HARMONIE.

## createNestInput.py
Creates the input for a DALES simulation with open boundary conditions from coarser DALES slab output. Used to nest DALES within itself.

## addCoordinates.py
Georeferences a DALES netcdf output file using the coordinate transformation given in the HARMONIE reference file. For both the coarse and fine simulation output. Used to visualize nested simulations with respect to each other.

## advecInt.py
Advective interpolation scheme to artificially increase the temporal resolution of the HARMONIE output.

## funcDALES.py
Some usefull functions used in multiple scripts.

## merge.sh
Merges DALES output files.

## H43_65lev.txt
Text file required to obtain the pressure levels of HARMONIE.
