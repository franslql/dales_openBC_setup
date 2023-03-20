#!/bin/bash
# Merge dales' files using cdo
# Merge fielddump files
rm -rf cdo
ln -s /perm/nmfl/cdo-2.0.5/src/cdo .
filename=$(ls fielddump.000.000.*)
if [ "${filename:0:9}" = "fielddump" ]; then
  # Find iexpnr number
  iexpnr=${filename:18:3}
  echo "Merge files for experiment ${iexpnr}"
  rm -r fielddump
  mkdir fielddump
  # Find number of y processes in x and y direction
  nprocx=$(ls "fielddump."*".000.${iexpnr}.nc" | wc -l | tr -d ' ')
  nprocy=$(ls fielddump.000.* | wc -l | tr -d ' ')
  nprocs=$(ls fielddump.* | wc -l | tr -d ' ')
  echo "Nprocx, Nprocy = ${nprocx}, ${nprocy}"
  # Find variable names
  varnames=$(./cdo -s showvar $filename)
  echo "Variables found: ${varnames}"
  # Loop over variables and merge them one by one
  for varname in $varnames
  do
    echo "Start merging ${varname}"
    ./cdo -f nc4 -z zip_6 -r -O collgrid,$nprocx,$varname `ls fielddump.* | sort -t y -k 3` "fielddump/${varname}.${iexpnr}.nc"
  done
fi
# Merge meancrosssection files
filename=$(ls meancrossxz.000.*)
if [ "${filename:0:11}" = "meancrossxz" ]; then
  # Find iexpnr number
  iexpnr=${filename:16:3}
  echo "Merge meancrossxz files for experiment ${iexpnr}"
  rm -r meancrossxz
  mkdir meancrossxz
  # Find number of processes in x direction
  nprocx=$(ls meancrossxz.* | wc -l | tr -d ' ')
  echo "Nprocx" = ${nprocx}
  # Find variable names
  varnames=$(./cdo -s showvar $filename)
  echo "Variables found: ${varnames}"
  for varname in $varnames
  do
    echo "Start merging ${varname}"
    ./cdo -f nc4 -z zip_6 -r -O collgrid,$nprocx,$varname `ls meancrossxz.* | sort -t y -k 3` "meancrossxz/${varname}mean.${iexpnr}.nc"
  done
fi
# Merge crosssection files xz
filename=$(ls crossxz.*.x000* | head -1)
if [ "${filename:0:7}" = "crossxz" ]; then
  # Find iexpnr number
  iexpnr=${filename:18:3}
  echo "Merge crossxz files for experiment ${iexpnr}"
  rm -r crossxz
  mkdir crossxz
  # Find number of processes in x direction
  level=${filename:8:4}
  nprocx=$(ls crossxz.${level}.x* | wc -l | tr -d ' ')
  echo "Nprocx" = ${nprocx}
  # Find variable names
  varnames=$(./cdo -s showvar $filename)
  echo "Variables found: ${varnames}"
  # Get all levels
  filenames=$(ls crossxz.*.x000*)
  for filename in $filenames
  do # Loop over all levels
    level=${filename:8:4}
    echo "Merge files for level ${level}"
    mkdir crossxz/${level}
    for varname in $varnames
    do
      echo "Start merging ${varname}"
      ./cdo -f nc4 -z zip_6 -r -O collgrid,$nprocx,$varname `ls crossxz.${level}.* | sort -t y -k 3` "crossxz/${level}/${varname}.${level}.${iexpnr}.nc"
    done
  done
fi
# Merge crosssection files yz
filename=$(ls crossyz.*.y000* | head -1)
if [ "${filename:0:7}" = "crossyz" ]; then
  # Find iexpnr number
  iexpnr=${filename:18:3}
  echo "Merge crossyz files for experiment ${iexpnr}"
  rm -r crossyz
  mkdir crossyz
  # Find number of processes in y direction
  level=${filename:8:4}
  nprocy=$(ls crossyz.${level}.y* | wc -l | tr -d ' ')
  echo "Nprocy" = ${nprocy}
  # Find variable names
  varnames=$(./cdo -s showvar $filename)
  echo "Variables found: ${varnames}"i
  # Get all levels
  filenames=$(ls crossyz.*.y000*)
  for filename in $filenames
  do # Loop over all levels
    level=${filename:8:4}
    echo "Merge files for level ${level}"
    mkdir crossyz/${level}
    for varname in $varnames
    do
      echo "Start merging ${varname}"
      ./cdo -f nc4 -z zip_6 -r -O collgrid,$nprocy,$varname `ls crossyz.${level}.* | sort -t y -k 3` "crossyz/${level}/${varname}.${level}.${iexpnr}.nc"
    done
  done
fi
# Merge xy files
filename=$(ls crossxy.*.x000y000* | head -1)
if [ "${filename:0:7}" = "crossxy" ]; then
  # Find iexpnr number
  iexpnr=${filename:22:3}
  echo "Merge crossxy files for experiment ${iexpnr}"
  rm -r crossxy
  mkdir crossxy
  # Find number of processes in x direction
  level=${filename:8:4}
  nprocx=$(ls crossxy.${level}.x*y000* | wc -l | tr -d ' ')
  echo "Nprocx" = ${nprocx}
  # Find variable names
  varnames=$(./cdo -s showvar $filename)
  echo "Variables found: ${varnames}"
  # Get all levels
  filenames=$(ls crossxy.*.x000y000*)
  for filename in $filenames
  do # Loop over levels
    level=${filename:8:4}
    echo "Merge files for level ${level}"
    mkdir crossxy/${level}
    for varname in $varnames
    do
      echo "Start merging ${varname}"
      ./cdo -f nc4 -z zip_6 -r -O collgrid,$nprocx,$varname `ls crossxy.${level}.* | sort -t y -k 3` "crossxy/${level}/${varname}.${level}.${iexpnr}.nc"
    done
  done
fi
# Merge cape files
filename=$(ls cape.x000y000*)
if [ "${filename:0:4}" = "cape" ]; then
  # Find iexpnr number
  iexpnr=${filename:14:3}
  echo "Merge cape files for experiment ${iexpnr}"
  rm -r cape
  mkdir cape
  # Find number of processes in y direction
  nprocy=$(ls cape.x000*y* | wc -l | tr -d ' ')
  echo "Nprocy" = ${nprocy}
  # Find variable names
  varnames=$(./cdo -s showvar $filename)
  echo "Variables found: ${varnames}"
  for varname in $varnames
  do
    echo "Start merging ${varname}"
    ./cdo -f nc4 -z zip_6 -r -O collgrid,$nprocy,$varname `ls cape.* | sort -t y -k 3` "cape/${varname}.${iexpnr}.nc"
  done
fi
