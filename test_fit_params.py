"""
Test NH3 fits vs. noise
=======================

Script to test cold_ammonia fits to NH3 (1,1) and (2,2) spectra,
given typical noise levels found in GBT Ammonia Survey (GAS)
"""
import numpy as np
import astropy.constants as con
import astropy.units as u
import astropy
from astropy.io import fits
import pyspeckit
from pyspeckit.spectrum.models import ammonia_constants, ammonia, ammonia_hf
from pyspeckit.spectrum.models.ammonia_constants import freq_dict
from pyspeckit.spectrum.units import SpectroscopicAxis, SpectroscopicAxes
import pylab as pl
import matplotlib.pyplot as plt
import copy

def get_mean_sdev(array,mask_value):
    array[array == mask_value] = np.nan
    array[array == 0] = np.nan
    mean = np.nanmean(array)
    sdev = np.nanstd(array)
    return mean, sdev

def gen_spec(model_params):
    """
    Generate a synthetic spectrum
    model_params = [Tkin, Tex, Ntot, width]
    """
    xarr11 = SpectroscopicAxis(np.linspace(-40,40,1000)*u.km/u.s,
                               velocity_convention='radio',
                               refX=freq_dict['oneone']).as_unit(u.GHz)
    xarr22 = SpectroscopicAxis(np.linspace(-40,40,1000)*u.km/u.s,
                               velocity_convention='radio',
                               refX=freq_dict['twotwo']).as_unit(u.GHz)
    xarr = SpectroscopicAxes([xarr11,xarr22])
    tkin = model_params[0]
    tex = model_params[1]
    ntot = model_params[2]
    width = model_params[3]
    fortho = 0.0
    synthspec = pyspeckit.spectrum.models.ammonia.cold_ammonia(xarr,tkin=tkin,tex=tex,ntot=ntot,
                                                               width=width,fortho=fortho)
    spectrum = pyspeckit.Spectrum(xarr=xarr,data=synthspec)
    return spectrum
             
def add_noise(spectrum,rms):
    """
    Add random noise to spectrum
    Can use error in Spectrum class
    rms is scale in K
    """
    noise = np.random.randn(spectrum.data.shape[0])*rms
    noisy_data = spectrum.data + noise
    noisy_spec = pyspeckit.Spectrum(xarr=spectrum.xarr,data=noisy_data)
    return noisy_spec

def run_many_fits(spectrum,rms,guesses,nruns):
    """
    guesses = [Tkin, Tex, Ntot, width, vlsr, fortho]
    """
    tk_fit    = []
    tex_fit   = []
    ntot_fit  = []
    width_fit = []
    for i in range(nruns):
        noisy_spectrum = add_noise(spectrum,rms)
        noisy_spectrum.specfit(fittype='cold_ammonia',guesses=guesses,fixed=[F,F,F,F,F,T])
        parcopy = copy.deepcopy(noisy_spectrum.specfit.parinfo)
        tk_fit = np.append(tk_fit,parcopy[0].value)
        tex_fit = np.append(tex_fit,parcopy[1].value)
        ntot_fit = np.append(ntot_fit,parcopy[2].value)
        width_fit = np.append(width_fit,parcopy[3].value)
    return tk_fit,tex_fit,ntot_fit,width_fit
            
       
def plot_over_tex(tk,width,rms,tex_arr,ntot_arr,guesses,nruns):
    """
    Run over arrays in Tex and Ntot (for now, must be 3 values each)
    Produce a 3x3 plot with contours over density for Tex vs. Ntot
    Return mean, standard deviation of fit parameters Tk, sigma
    This takes forever, should write out the results so don't have to run it again. 
    """
    # Set up plot
    gridsize = 50
    fig,axes = plt.subplots(3,3,figsize=(8,8))
    axes = axes.ravel()
    i=0
    results = {}
    # Generate spectrum
    for tex in tex_arr:
        for ntot in ntot_arr:
            model_params = [tk,tex,ntot,width]
            spectrum = gen_spec(model_params)
            tk_fit, tex_fit, ntot_fit, width_fit = run_many_fits(spectrum,rms,guesses,nruns)
            ax = axes[i]
            im = ax.hexbin(tex_fit,ntot_fit,gridsize=gridsize,cmap=plt.cm.pink_r,
                           extent=[2,10,13.3,14.8])
            ax.set_ylabel('log $N$(NH$_3$)')
            ax.set_xlabel('$T_\mathrm{ex}$ (K)')
            ax.tick_params(axis='both',which='major',labelsize=10)
            ax.text(9,14.8,'$T_\mathrm{ex} = {0} K'.format(tex),
                    horizontalalignment='right',fontsize=8)
            ax.text(9,14.6,'log $N$(NH$_3$) = {0} cm$^{-2}$'.format(ntot),
                    horizontalalignment='right',fontsize=8)
            cb = fig.colorbar(im,ax=ax) 
            cb.set_label('N',fontsize=10)
            cb.ax.tick_params(labelsize=9)
            # Calculate output mean, standard deviations, number of failed runs:
            mean_tk, sdev_tk = get_mean_sdev(tk_fit,2.7315)
            mean_tex, sdev_tex = get_mean_sdev(tex_fit,2.7315)
            mean_ntot, sdev_ntot = get_mean_sdev(ntot_fit,13.)
            mean_width, sdev_width = get_mean_sdev(width_fit,0.)
            # Add results to dictionary:
            output_dict = {'InpTex':tex,'InpTk':tk,'InpNtot':ntot,'InpWidth':width,
                           'meanTk':mean_tk,'sdevTk':sdev_tk,
                           'meanTex':mean_tex,'sdevTex':sdev_tex,
                           'meanNtot':mean_ntot,'sdevNtot':sdev_ntot,
                           'meanWidth':mean_width,'sdevWidth':sdev_width}
            results['run{0}'.format(i)] = output_dict
            i = i+1

    plt.tight_layout()
    fig.savefig('figures/test_fit_params_panel.pdf')
    plt.close('all')
    return results


# Parameters
tkin = 10.0
tex = 4.
ntot = 14.0
width = 0.25
model_params = [tkin,tex,ntot,width]
# Fit setup
F = False
T = True   
                   
rms = 0.1

tex_arr = [3.5,4.5,5.5]
ntot_arr = [13.8,14.2,14.6]
guesses = [12,3,14.5,0.5,0,0]
nruns = 500

results = plot_over_tex(tkin,width,rms,tex_arr,ntot_arr,guesses,nruns)

'''
# Generate NH3 (1,1) and (2,2) spectra
spectrum = gen_spec(model_params)

# Add vlsr, fortho to guesses
guesses = [10,5,14,0.3,0,0]
#guesses.extend([0,0])
nruns = 500
tk_fit,tex_fit,ntot_fit,width_fit = run_many_fits(spectrum,rms,guesses,nruns)

# Plot results
gridsize=25
fig,axes = plt.subplots(3,2,figsize=(6,5))
axes = axes.ravel()
ax0 = axes[0]
im0 = ax0.hexbin(tex_fit,ntot_fit,gridsize=gridsize,cmap=plt.cm.pink_r,extent=[2,10,ntot-0.6,ntot+0.6])
ax0.set_ylabel('log $N$(NH$_3$)')
ax0.set_xlabel('$T_\mathrm{ex}$ (K)')
ax0.tick_params(axis='both',which='major',labelsize=11)
cb0 = fig.colorbar(im0,ax=ax0)
cb0.set_label('N',fontsize=11)
cb0.ax.tick_params(labelsize=10)
ax1 = axes[1]
im1 = ax1.hexbin(tex_fit,tk_fit,gridsize=gridsize,cmap=plt.cm.pink_r,extent=[2,10,3,20])
ax1.set_ylabel('$T_\mathrm{K}$ (K)')
ax1.set_xlabel('$T_\mathrm{ex}$ (K)')
ax1.tick_params(axis='both',which='major',labelsize=11)
cb1 = fig.colorbar(im1,ax=ax1)
cb1.set_label('N',fontsize=11)
cb1.ax.tick_params(labelsize=10)
ax2 = axes[2]
ax2.hist(tex_fit,30,histtype='step',range=(2,10))
ax2.tick_params(axis='both',which='major',labelsize=11)
ax2.set_xlabel('$T_\mathrm{ex}$ (K)',fontsize=11)
ax3 = axes[3]
ax3.hist(ntot_fit,20,histtype='step',range=(ntot-0.6,ntot+0.6))
ax3.tick_params(axis='both',which='major',labelsize=11)
ax3.set_xlabel('log $N$(NH$_3$)',fontsize=11)
ax4 = axes[4]
ax4.hist(tk_fit,30,histtype='step',range=(3,20))
ax4.tick_params(axis='both',which='major',labelsize=11)
ax4.set_xlabel('$T_\mathrm{K}$ (K)',fontsize=11)
ax5 = axes[5]
ax5.hist(width_fit,30,histtype='step',range=(0.05,0.5))
ax5.tick_params(axis='both',which='major',labelsize=11)
ax5.set_xlabel('$\sigma_v$ (km s$^{-1}$)',fontsize=11)
plt.tight_layout()
fig.savefig('figures/test_fit_params.pdf')
plt.close('all')

# Don't know how to plot spectrum using NH3 parameters (i.e., two separate windows)
# without fit attached
# Plot one noisy spectrum for show:
noisy_spectrum = add_noise(spectrum,rms)
noisy_spectrum.specfit(fittype='cold_ammonia',guesses=guesses,fixed=[F,F,F,F,F,T])

import types
noisy_spectrum.plot_special = types.MethodType(pyspeckit.wrappers.fitnh3.plotter_override,
                                               noisy_spectrum)
fig = plt.figure(figsize=(4,6))
noisy_spectrum.plot_special(vrange=[-30,30])
plt.tight_layout()
fig.savefig('figures/test_model_spectrum.pdf')
plt.close('all')
'''
