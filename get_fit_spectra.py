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

def plot_11_22_fit(cubes,coords,region):
    # Set up plot parameters
    fig = pl.figure(1,figsize=(4,6))
    xlim = [-22,30]
    ylim = [-0.4,2.0]
    # (1,1) line first
    cubes.xarr.refX = pyspeckit.spectrum.models.ammonia.freq_dict['oneone']
    cubes.xarr.refX_unit='Hz'
    cubes.xarr.velocity_convention='radio'
    cubes.xarr.convert_to_unit('km/s')
    ax = fig.add_subplot(2,1,1)
    cubes.plotter(axis=ax)
    cubes.plot_spectrum(coords[0],coords[1],plot_fit=True)
    cubes.plotter.axis.set_xlim(xlim[0],xlim[1])
    cubes.plotter.axis.set_ylim(ylim[0],ylim[1])
    # (2,2) line next
    # Convert cube back to Hz and then to km/s using (2,2) rest frequency
    # MUST BE A BETTER WAY 
    cubes.xarr.convert_to_unit('Hz')
    cubes.xarr.refX = pyspeckit.spectrum.models.ammonia.freq_dict['twotwo']
    cubes.xarr.refX_unit='Hz'
    cubes.xarr.velocity_convention='radio'
    cubes.xarr.convert_to_unit('km/s')
    # Set up subplot
    ax = fig.add_subplot(2,1,2)
    cubes.plotter(axis=ax)
    cubes.plot_spectrum(coords[0],coords[1],plot_fit=False)
    cubes.plot_fit(coords[0],coords[1],annotate=False)
    cubes.plotter.axis.set_xlim(xlim[0],xlim[1])
    cubes.plotter.savefig('figures/{0}_spectrum.png'.format(region),bbox_inches='tight')


# NH3 (1,1) cube
region = 'L1688'
file_extension = 'DR1_rebase3'
line = 'NH3_11'
OneOneFile = '../testing/{0}/{0}_NH3_11_{1}_trim.fits'.format(region,file_extension)
TwoTwoFile = '../testing/{0}/{0}_NH3_22_{1}_trim.fits'.format(region,file_extension)
FitFile = '../testing/{0}/{0}_parameter_maps_{1}_trim.fits'.format(region,file_extension)

cube11 = pyspeckit.Cube(OneOneFile)
cube11.unit = 'K'
cube22 = pyspeckit.Cube(TwoTwoFile)
cube22.unit = 'K'
cubes = pyspeckit.CubeStack([cube11,cube22])
cubes.unit = 'K'

if not 'cold_ammonia' in cubes.specfit.Registry.multifitters:
    cubes.specfit.Registry.add_fitter('cold_ammonia',ammonia.cold_ammonia_model(),6)

cubes.load_model_fit(FitFile,6,npeaks=1,fittype='cold_ammonia')
cubes.specfit.parinfo[5]['fixed'] = True

# This gives an AssertionError
#cubes.plot_special = pyspeckit.wrappers.fitnh3.plotter_override
#cubes.plot_special_kwargs = {'fignum':2}

pix_coords = [54,184]

plot_11_22_fit(cubes,pix_coords,region)
