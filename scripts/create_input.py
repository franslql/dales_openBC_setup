#%% Load modules
import json
from GridDales import GridDales
from prep_harmonie import prep_harmonie
from initial_fields import initial_fields
from boundary_fields import boundary_fields
from profiles import profiles
from surface_temperature import surface_temperature
from synthetic_turbulence import synthetic_turbulence
from gaussian_filter import gaussian_filter
#%% Read input file
with open('input.json') as f: input = json.load(f)
#%% Create input for outer simulation
if 'coarse' in input:
  input_coarse = input['coarse']
  input_coarse['author'] = input['author']
  #%% Create DALES grid
  grid = GridDales(input_coarse['grid'])
  #%% Transfor input data to rectilinear grid and to prognostic variables of DALES
  if(input_coarse['source'].lower() == 'harmonie'):
    data,transform = prep_harmonie(input_coarse,grid)
  else:
    print('unvalid source type')
    exit()
  #%% Apply spatial horizontal Gaussian filter to data
  if('filter' in input_coarse):
    data = gaussian_filter(data,input_coarse)
  #%% Advective time interpolation of input data (optional, to be implemented)
  
  #%% Create initial fields > initfields.inp.xxx.nc
  initfields = initial_fields(input_coarse,grid,data,transform)
  print('finished initifields')
  #%% Create profiles > prof.inp.xxx, lscale.inp.xxx scalar.inp.xxx
  profiles(input_coarse,grid,initfields,data)
  print('finished profiles')
  #%% Create boundary input > openboundaries.inp.xxx.nc
  openboundaries = boundary_fields(input_coarse,grid,data)
  print('finished boundary fields')
  #%% Create synthetic turbulence for boundary input (optional) > openboundaries.inp.xxx.nc
  if('synturb' in input_coarse): 
    synturb = synthetic_turbulence(input_coarse,grid,data,transform)
    print('finished synturb')
  #%% Create heterogeneous and time dependend skin temperature > tskin.inp.xxx.nc (if ltskin==true)
  if('tskin' in input_coarse): 
    tskin = surface_temperature(input_coarse,grid,data,transform)
    print('finished tskin')

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