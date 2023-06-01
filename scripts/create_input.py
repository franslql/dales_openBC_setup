#%% Load modules
import json
from GridDales import GridDales
from prep_harmonie import prep_harmonie
from initial_fields import initial_fields
from boundary_fields import boundary_fields
from profiles import profiles
from surface_temperature import surface_temperature
from synthetic_turbulence import synthetic_turbulence
#%% Read input file
with open('input.json') as f: input = json.load(f)
#%% Create input for outer simulation
if 'coarse' in input:
  input_coarse = input['coarse']
  input_coarse['author'] = input['author']
  #%% Create DALES grid
  grid = GridDales(input_coarse)
  #%% Transfor input data to rectilinear grid and to prognostic variables of DALES
  import importlib
  import prep_harmonie
  importlib.reload(prep_harmonie)
  from prep_harmonie import prep_harmonie
  if(input_coarse['source'].lower() == 'harmonie'): 
    data,transform = prep_harmonie(input_coarse,grid)
  else:
    print('unvalid source type')
    exit()
  #%% Advective time interpolation of input data (optional, to be implemented)
  
  #%% Create initial fields > initfields.inp.xxx.nc
  initfields = initial_fields(input_coarse,grid,data,transform)
  #%% Create profiles > prof.inp.xxx, lscale.inp.xxx scalar.inp.xxx
  profiles(input_coarse,grid,initfields,data)
  #%% Create boundary input > openboundaries.inp.xxx.nc
  openboundaries = boundary_fields(input_coarse,grid,data)
  #%% Create synthetic turbulence for boundary input (optional) > openboundaries.inp.xxx.nc
  import importlib
  import synthetic_turbulence
  importlib.reload(synthetic_turbulence)
  from synthetic_turbulence import synthetic_turbulence
  if(input_coarse['lsynturb']): 
    synturb = synthetic_turbulence(input_coarse,grid,data,transform)
  #%% Create heterogeneous and time dependend skin temperature > tskin.inp.xxx.nc (if ltskin==true)
  if(input_coarse['ltskininp']): 
    tskin = surface_temperature(input_coarse,grid,data,transform)

#%% Write data to input files
if('fine' in input):
  input_fine = input['fine']
  input_fine['author'] = input['author']
  # Create DALES grid
  grid = GridDales(input_fine)  
  # Create initial fields > initfields.inp.xxx.nc

  # Create boundary input > openboundaries.inp.xxx.nc

  # Create heterogeneous and time dependend skin temperature > tskin.inp.xxx.nc
# %%
