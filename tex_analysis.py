from astropy.io import fits
import aplpy
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as c
import warnings
import numpy as np
import os
import FITS_tools
import matplotlib.patches as patches
from scipy.ndimage import binary_opening
from abundance_analysis_config import plottingDictionary
from matplotlib.ticker import AutoMinorLocator

def plot_tex_etex_panel(tex_file,etex_file,eTex_lim,nh3_22_file,nh3_22_rms,region,plot_param,ax):
    tex_data = fits.getdata(tex_file)
    # Mask where uncertainties on Tex are high
    eTex_data = fits.getdata(etex_file)
    tex_data[(eTex_data > eTex_lim)] = np.nan
    # Mask where no good fits (data == 0)
    tex_data[(tex_data == 0)] = np.nan
    # Highlight where low S/N in (2,2) line:
    nh3_22_data = fits.getdata(nh3_22_file)
    rms_22 = fits.getdata(nh3_22_rms)
    snr = nh3_22_data/rms_22
    tex_data[snr < plot_param['snr_lim']] = np.nan
    
    gridsize = plot_param['temp_grid_size']
    finite_index = np.where(np.isfinite(tex_data))
    # NH3 abundance already in log
    # Use hexbin, scatter plots not useful here
    im = ax.hexbin(tex_data[finite_index],
                   eTex_data[finite_index],
                   gridsize=gridsize,cmap=plt.cm.pink_r,bins='log',
                   extent=[2,16,0,5])
    #ax.scatter(tex_data[finite_index],nh3eTex[finite_index],s=2,c='black')
    #ax.scatter(tex_data[snr < 3.], eTex_data[snr < 3.], s=0.75, c='red',edgecolors='face',alpha=0.3)
    ax.plot([plot_param['tex_lim'],plot_param['tex_lim']],[0,50],color='red')
    ax.set_ylabel('$\Delta$ T$_{ex}$ (K)')
    ax.set_xlabel('T$_{ex}$ (K)')
    ax.set_ylim(0,5)
    ax.set_xlim(2,16)
    ax.tick_params(axis='both',which='major',labelsize=11)
    ax.text(0.92,0.82,'{0}'.format(region),fontsize=10,horizontalalignment='right',
            transform=ax.transAxes)
    cb = fig.colorbar(im,ax=ax,ticks=[0,0.5,1,1.5,2,2.5])
    cb.set_label('log10(N)',fontsize=11)
    cb.ax.tick_params(labelsize=10)


def plot_tex_col(tex_file,h2_col_file,nh3_etex_file,eTex_lim,region,plot_param):
    h2_col_data = fits.getdata(h2_col_file)
    tex_data = fits.getdata(tex_file)
    # Mask where uncertainties on Tex are high
    nh3eTex = fits.getdata(nh3_etex_file)
    tex_data[(nh3eTex > eTex_lim)] = np.nan
    # Mask where no good fits (data == 0)
    tex_data[(tex_data == 0)] = np.nan

    gridsize = plot_param['nh2_grid_size']
    finite_index = np.where(np.isfinite(tex_data))
    fig = plt.figure(figsize=(5,3.5))
    ax = plt.gca()
    # NH3 abundance already in log
    # Use hexbin, scatter plots not useful here
    plt.hexbin(np.log10(h2_col_data[finite_index]),
               tex_data[finite_index],
               gridsize=gridsize,cmap=plt.cm.pink_r,bins='log',
               extent=[plot_param['h2_col_lim'][0],plot_param['h2_col_lim'][1],
                       3,15])
    ax.set_ylabel('T$_{ex}$ (K)')
    ax.set_xlabel('log $N$(H$_2$) (cm$^{-2}$)')
    plt.figtext(0.2,0.85,'{0}'.format(region),fontsize=14)
    cb = plt.colorbar()
    cb.set_label('log10(N)')
    plt.legend(loc=2,frameon=False)
    plt.tight_layout()
    fig.savefig('figures/{0}_NH2_vs_Tex.pdf'.format(region),)
    plt.close('all')

def plot_n_nh3_mom0_panel(n_nh3_file, mom0file, tex_file, etex_file, eTex_lim,
                          region,plot_param,ax):
    nnh3_data = fits.getdata(n_nh3_file)
    mom0_data = fits.getdata(mom0file)
    tex_data = fits.getdata(tex_file)
    # Mask where uncertainties on Tex are high
    nh3eTex = fits.getdata(etex_file)
    nnh3_data[(nh3eTex > eTex_lim)] = np.nan
    # Mask where no good fits (data == 0)
    nnh3_data[(nnh3_data == 0)] = np.nan
    # Get where Tex < tex_lim
    low_tex = np.where(tex_data < plot_param['tex_lim'])
    gridsize = plot_param['temp_grid_size']
    finite_index = np.where(np.isfinite(tex_data))
    # NH3 abundance already in log
    # Use hexbin, scatter plots not useful here
    #im = ax.hexbin(tex_data[finite_index],
    #               nh3eTex[finite_index],
    #               gridsize=gridsize,cmap=plt.cm.pink_r,bins='log',
    #               extent=[2,16,0,5])
    ax.scatter(np.log10(mom0_data[finite_index]),nnh3_data[finite_index],s=0.5,c='gray')
    ax.scatter(np.log10(mom0_data[low_tex]),nnh3_data[low_tex],s=2,c='r',
               edgecolors='face',alpha=0.5,zorder=2,
               label='T$_{{ex}}$ < {:.1f} K'.format(plot_param['tex_lim']))
    #ax.plot([4,4],[0,50],color='red')
    ax.set_ylabel('log $N$(NH$_3$) (cm$^{-2}$)')
    ax.set_xlabel('log mom0 NH$_3$ (K km s$^{-1}$)')
    ax.set_ylim(13,16)
    #ax.set_xlim(-0.5,1.6)
    ax.set_xlim(-1.5,1.5)
    ax.set_xticks([-0.5,0,0.5,1,1.5])
    ax.set_yticks([13,14,15,16])
    minor_locator = AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.yaxis.set_minor_locator(minor_locator)    
    ax.tick_params(axis='both',which='major',labelsize=11)
    ax.text(0.92,0.1,'{0}'.format(region),fontsize=10,horizontalalignment='right',
            transform=ax.transAxes)
    ax.legend(loc=1,frameon=False,fontsize=10,scatterpoints=1)
    #cb = fig.colorbar(im,ax=ax,ticks=[0,0.5,1,1.5,2,2.5])
    #cb.set_label('log10(N)',fontsize=11)
    #cb.ax.tick_params(labelsize=10)


def plot_tex_etex_tk_panel(tex_file,etex_file,eTex_lim,Tk_file,eTk_file,etk_lim, 
                           region,plot_param,ax):
    tex_data = fits.getdata(tex_file)
    # Mask where uncertainties on Tex are high
    nh3eTex = fits.getdata(etex_file)
    tex_data[(nh3eTex > eTex_lim)] = np.nan
    # Look at Tk, eTk:
    tk_data  = fits.getdata(Tk_file)
    etk_data = fits.getdata(eTk_file)
    etk_data[np.isnan(tex_data)] = np.nan
    # Mask where no good fits (Tk == 0)
    tex_data[(tk_data == 0)] = np.nan
    tex_data[(etk_data > etk_lim)] = np.nan

    gridsize = plot_param['temp_grid_size']
    finite_index = np.where(np.isfinite(tex_data))
    # NH3 abundance already in log
    # Use hexbin, scatter plots not useful here
    im = ax.hexbin(tex_data[finite_index],
                   nh3eTex[finite_index],
                   gridsize=gridsize,cmap=plt.cm.pink_r,bins='log',
                   extent=[2,16,0,5])
    #ax.scatter(tex_data[finite_index],nh3eTex[finite_index],s=2,c='black')
    #ax.scatter(tex_data[errRatio > 0.1], nh3eTex[errRatio > 0.1], s=0.75, c='red',edgecolors='face',alpha=0.3)
    ax.plot([plot_param['tex_lim'],plot_param['tex_lim']],[0,50],color='red')
    ax.set_ylabel('$\Delta$ T$_{ex}$ (K)')
    ax.set_xlabel('T$_{ex}$ (K)')
    ax.set_ylim(0,5)
    ax.set_xlim(2,16)
    ax.tick_params(axis='both',which='major',labelsize=11)
    ax.text(0.92,0.82,'{0}'.format(region),fontsize=10,horizontalalignment='right',
            transform=ax.transAxes)
    cb = fig.colorbar(im,ax=ax,ticks=[0,0.5,1,1.5,2,2.5])
    cb.set_label('log10(N)',fontsize=11)
    cb.ax.tick_params(labelsize=10)


herschel_nh2_dir = 'nh2_regridded'
herschel_td_dir  = 'Td_regridded'
herschel_im_dir  = 'h500_regridded'
region_list  = ['B18','NGC1333','L1688','OrionA']
file_extension = 'DR1_rebase3'
# Stick with DR1 or use more extended regions? L1688 is bigger, for example. 
# Set uncertainty limit on Tex for good N(NH3) fits
eTex_lim = 5 # K
eTk_lim = 2. # K

for region in region_list:
    plot_param = plottingDictionary[region]
    if region in ['OrionA']:
        herschelNH2File = '{0}/{1}_NH2_regrid_masked.fits'.format(herschel_nh2_dir,region)
    else:
        herschelNH2File = '{0}/{1}_NH2_regrid.fits'.format(herschel_nh2_dir,region)
    nh3TexFits = 'propertyMaps/{0}_Tex_{1}_flag.fits'.format(region,file_extension)
    nh3eTexFits = 'propertyMaps/{0}_eTex_{1}_flag.fits'.format(region,file_extension)
    plot_tex_col(nh3TexFits,herschelNH2File,nh3eTexFits,eTex_lim,region,plot_param)

# Set up for panel plot
fig,axes = plt.subplots(2,2,figsize=(7.5,5))
axes = axes.ravel()

for i in range(len(region_list)):
    region = region_list[i]
    plot_param = plottingDictionary[region]
    nh3TexFits = 'propertyMaps/{0}_Tex_{1}_flag.fits'.format(region,file_extension)
    nh3eTexFits = 'propertyMaps/{0}_eTex_{1}_flag.fits'.format(region,file_extension)
    nh322Fits = 'nh3_data/{0}_NH3_22_{1}_mom0_QA_trim.fits'.format(region,file_extension)
    nh322RmsFits = 'nh3_data/{0}_NH3_22_{1}_rms_QA_trim.fits'.format(region,file_extension)    
    ax = axes[i]
    plot_tex_etex_panel(nh3TexFits,nh3eTexFits,eTex_lim,
                        nh322Fits,nh322RmsFits,
                        region,plot_param,ax)

plt.tight_layout()
fig.savefig('figures/DR1_Tex_eTex_snr.pdf')
plt.close('all')

# Set up for panel plot
fig,axes = plt.subplots(2,2,figsize=(7.5,5))
axes = axes.ravel()

for i in range(len(region_list)):
    region = region_list[i]
    plot_param = plottingDictionary[region]
    nnh3_file = 'propertyMaps/{0}_N_NH3_{1}_flag.fits'.format(region,file_extension)
    mom0_file = 'nh3_data/{0}_NH3_22_{1}_mom0_QA_trim.fits'.format(region,file_extension)
    nh3TexFits = 'propertyMaps/{0}_Tex_{1}_flag.fits'.format(region,file_extension)
    nh3eTexFits = 'propertyMaps/{0}_eTex_{1}_flag.fits'.format(region,file_extension)
    ax = axes[i]
    plot_n_nh3_mom0_panel(nnh3_file, mom0_file, nh3TexFits, nh3eTexFits, eTex_lim,
                          region,plot_param,ax)

plt.tight_layout()
fig.savefig('figures/DR1_N_NH3_mom0.pdf')
plt.close('all')

# Set up for panel plot
fig,axes = plt.subplots(2,2,figsize=(7.5,5))
axes = axes.ravel()

for i in range(len(region_list)):
    region = region_list[i]
    plot_param = plottingDictionary[region]
    nh3TexFits = 'propertyMaps/{0}_Tex_{1}_flag.fits'.format(region,file_extension)
    nh3eTexFits = 'propertyMaps/{0}_eTex_{1}_flag.fits'.format(region,file_extension)
    nh3TkFits   = 'propertyMaps/{0}_Tkin_{1}_flag.fits'.format(region,file_extension)
    nh3eTkFits = 'propertyMaps/{0}_eTkin_{1}_flag.fits'.format(region,file_extension)
    ax = axes[i]
    plot_tex_etex_tk_panel(nh3TexFits,nh3eTexFits,eTex_lim,nh3TkFits,nh3eTkFits,eTk_lim,
                           region,plot_param,ax)

plt.tight_layout()
fig.savefig('figures/DR1_Tex_eTex_Tk.pdf')
plt.close('all')
