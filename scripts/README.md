# scripts

## create_input.py
Main script to create DALES input files. Reads input.json and calls the other scripts.

## GridDALES.py
Called by create_input.py. Creates the DALES grid.

## prep_harmonie.py
Called by create_input.py. Prepares HARMONIE fields to be used by boundary_fields.py, initial_fields.py, profiles.py and synthetic_turbulence.py. Creates a coordinate transform from a WGS84 grid to a local rectilinear grid used by DALES.

## initial_fields.py
Called by create_input.py. Interpolates the initial fields to the DALES grid and creates initfields.inp.xxx.nc.

## profiles.py
Called by create_input.py. Creates prof.inp.xxx by averaging the initial fields in initfields.inp.xxx.nc. Creates scalar.inp.xxx, set to zeros. Creates lscale.inp.xxx, set to zeros. Creates exnr.inp.xxx which was used in prep_harmonie.py.

## boundary_fields.py
Called by create_input.py. Interpolates the fields to the boundaries of the DALES grid and creates openboundaries.inp.xxx.nc.

## synthetic_turbulence.py
Called by create_input.py. Creates the input for the synthetic turbulence routine and adds it to openboundaries.inp.xxx.nc.

## surface_temperature.py
Called by create_input.py. Interpolates the era5 skin temperature to the DALES grid and creates tskin.inp.xxx.nc.

## input.json
Used by create_input.py. Example script to create input for a simulation 1-way nested in HARMONIE.

## H43_65lev.txt
Used by prep_harmonie.py. File containg HARMONIE level information.

# merge.sh
Bash script to merge DALES output files.

## old_scripts
Folder containing older scripts used to create boundary input.
