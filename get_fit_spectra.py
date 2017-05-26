import pyspeckit
import astropy
import astropy.io.fits as fits
import numpy as np
import astropy.constants as con
import astropy.units as u
import os
from astropy.convolution import convolve_fft,Gaussian2DKernel
from spectral_cube import SpectralCube
from pyspeckit.spectrum.models import ammonia
import matplotlib.pyplot as plt
import pylab as pl

# NH3 (1,1) cube
region = 'L1688'
file_extension = 'DR1_rebase3'
line = 'NH3_11'
OneOneFile = 'nh3_data/{0}_NH3_11_{1}_trim.fits'.format(region,file_extension)
TwoTwoFile = 'nh3_data/{0}_NH3_22_{1}_trim.fits'.format(region,file_extension)
FitFile = 'propertyMaps/{0}_parameter_maps_{1}_flag.fits'.format(region,file_extension)

cube11 = pyspeckit.Cube(OneOneFile)
cube11.unit = 'K'
cube22 = pyspeckit.Cube(TwoTwoFile)
cube22.unit = 'K'
cubes = pyspeckit.CubeStack([cube11,cube22])
cubes.unit = 'K'

if not 'cold_ammonia' in cubes.specfit.Registry.multifitters:
    cubes.specfit.Registry.add_fitter('cold_ammonia',ammonia.cold_ammonia_model(),6)

#cubes.load_model_fit(FitFile,6,npeaks=1,fittype='cold_ammonia')
#cubes.specfit.parinfo[5]['fixed'] = True

pix_coords = [54,184]

sp = cubes.get_spectrum(pix_coords[0],pix_coords[1])
F = False
T = True
sp.specfit(fittype='cold_ammonia',guesses=[10,3,14,0.3,3.33,0],fixed=[F,F,F,F,F,T])
#sp.specfit(fittype='cold_ammonia',guesses=[8,3,14,0.24,3.33,0],fixed=[F,F,F,F,F,T])

# Set up plot parameters
fig = plt.figure(figsize=(4,6))
xlim = [-22,30]
ylim = [-0.4,2.0]

spdict = pyspeckit.wrappers.fitnh3.BigSpectrum_to_NH3dict(sp,vrange=[-30,30])
pyspeckit.wrappers.fitnh3.plot_nh3(spdict,sp)

plt.tight_layout()

fig.savefig('figures/test_spectrum.pdf')
plt.close('all')
#plot_11_22_fit(cubes,pix_coords,region)
