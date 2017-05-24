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
from temp_analysis_config import plottingDictionary

def get_prot_loc(region):
    protFile = 'protostars/{0}_protostar_list.txt'.format(region)
    ra, dec = np.loadtxt(protFile,unpack=True)
    return ra, dec

def log10_h2(h2_image):
    log_h2_data = np.log10(h2_image.data)
    log_h2_hdu = fits.PrimaryHDU(log_h2_data,h2_image.header)
    return log_h2_hdu    

#def plot_temp_compare(nh3_tk_file,h2_td_file,snr,snr_lim,region,plot_param):
def plot_temp_compare(nh3_tk_file,h2_td_file,eTk_file,eTk_lim,region,plot_param):
    eTk_data = fits.getdata(eTk_file)
    nh3_temp_data = fits.getdata(nh3_tk_file)
    h2_temp_data = fits.getdata(h2_td_file)
    # Mask out pixels with large errors in Tk
    nh3_temp_data[eTk_data > eTk_lim] = np.NaN
    #nh3_temp_data[snr < snr_lim] = np.NaN
    nh3_temp_data[nh3_temp_data == 0] = np.NaN
    maxTkin = plot_param['nh3_temp_lim'][1]
    finite_index = np.where(np.logical_and(np.isfinite(nh3_temp_data),nh3_temp_data < maxTkin))
    fig = plt.figure()
    ax = plt.gca()
    # Plot temperatures
    plt.hexbin(nh3_temp_data[finite_index],h2_temp_data[finite_index],
               gridsize=plot_param['temp_grid_size'],bins='log',cmap=plt.cm.pink_r,
               extent=[plot_param['nh3_temp_lim'][0],plot_param['nh3_temp_lim'][1],
                       plot_param['her_temp_lim'][0],plot_param['her_temp_lim'][1]])
    ax.set_xlabel(r'$T_\mathrm{kin}$ (K)')
    ax.set_ylabel(r'$T_\mathrm{dust}$ (K)')
    plt.figtext(0.15,0.8,'{0}'.format(region),fontsize=14)
    equality_pts = [np.max([plot_param['nh3_temp_lim'][0],plot_param['her_temp_lim'][0]]),
                    np.min([plot_param['nh3_temp_lim'][1],plot_param['her_temp_lim'][1]])]
    ax.plot(equality_pts,equality_pts,linewidth=3,color='gray',alpha=0.6,label='$T_d = T_K$')
    cb = plt.colorbar()
    cb.set_label('log10(N)')
    plt.legend(loc=2,frameon=False)
    fig.savefig('figures/{0}_Herschel_vs_NH3_temp.pdf'.format(region))
    plt.close('all')

def plot_temp_overlay(nh3_temp_file,h2_temp_file,nh3_cont_fits,region,plot_param,include_prot=True):
    nh3_temp_hdu = fits.open(nh3_temp_file)
    h2_temp_hdu  = fits.open(h2_temp_file)
    temp_ratio = nh3_temp_hdu[0].data/h2_temp_hdu[0].data
    temp_ratio[np.where(temp_ratio == 0)] = np.nan
    temp_ratio_hdu = fits.PrimaryHDU(temp_ratio,nh3_temp_hdu[0].header)
    text_size = 14
    b18_text_size = 20    
    # Contour parameters (currently NH3 moment 0)
    cont_color='indigo'
    cont_lw   = 0.6
    cont_levs=2**np.arange( 0,20)*plot_param['w11_step']
    fig=aplpy.FITSFigure(temp_ratio_hdu,figsize=(plot_param['size_x'], plot_param['size_y']))
    fig.show_colorscale(cmap='afmhot',vmin=0.5,vmax=1.5)
    fig.set_nan_color('0.95')
    # Observations mask contour
    #fig.show_contour(obsMaskFits,colors='black',levels=1,linewidths=1.5)
    # NH3 moment contours
    # Masking of small (noisy) regions
    selem = np.array([[0,1,0],[1,1,1],[0,1,0]])
    LowestContour = cont_levs[0]*0.5
    w11_hdu = fits.open(nh3_cont_fits)
    map = w11_hdu[0].data
    mask = binary_opening(map > LowestContour, selem)
    MaskedMap = mask*map
    w11_hdu[0].data = MaskedMap
    # Labels
    if region == 'B18':
        text_size = b18_text_size
    label_colour = 'black'
    fig.show_contour(w11_hdu,colors=cont_color,levels=cont_levs,linewidths=cont_lw)
    # Ticks
    fig.ticks.set_color('black')
    fig.tick_labels.set_xformat('hh:mm:ss')
    fig.tick_labels.set_style('colons')
    fig.tick_labels.set_yformat('dd:mm')
    # Scale bar
    ang_sep = (plot_param['scalebar_size'].to(u.au)/plot_param['distance']).to(u.arcsec, equivalencies=u.dimensionless_angles())
    fig.add_colorbar()
    fig.colorbar.set_width(0.15)
    fig.colorbar.show(box_orientation='horizontal', width=0.1, pad=0.0, location='top')
    fig.colorbar.set_font(family='sans_serif',size=text_size)
    fig.add_scalebar(ang_sep.to(u.degree))
    fig.scalebar.set_corner(plot_param['scalebar_pos'])
    fig.scalebar.set(color='black')
    fig.scalebar.set_label('{0:4.2f}'.format(plot_param['scalebar_size']))
    fig.scalebar.set_font(family='sans_serif',size=text_size)
    fig.tick_labels.set_font(family='sans_serif',size=text_size)
    fig.axis_labels.set_font(family='sans_serif',size=text_size)
    fig.add_label(plot_param['label_xpos'], plot_param['label_ypos'], 
                  '{0}\n{1}'.format(region,'$T_K \ / \ T_d$'), 
                  relative=True, color=label_colour, 
                  horizontalalignment=plot_param['label_align'],
                  family='sans_serif',size=text_size)
    if include_prot:
        ra_prot, de_prot = get_prot_loc(region)
        if region == 'B18':
            marker_size = 70
        else: 
            marker_size = 50
        fig.show_markers(ra_prot,de_prot,marker='*',s=marker_size,c='white',edgecolor='black',linewidth=cont_lw*1.5,zorder=4)
        fig.save('figures/{0}_tempRatio_image_prot.pdf'.format(region),adjust_bbox=True,dpi=120)
    else:
        fig.save( 'figures/{0}_tempRatio_image.pdf'.format(region),adjust_bbox=True,dpi=120)#, bbox_inches='tight')
    fig.close()    


def plot_temp_h2(nh3_tk_file,h2_td_file,h2_col_file,eTk_file,eTk_lim,region,plot_param):
    eTk_data = fits.getdata(eTk_file)
    nh3_temp_data = fits.getdata(nh3_tk_file)
    h2_temp_data = fits.getdata(h2_td_file)
    h2_col_data = fits.getdata(h2_col_file)
    # Mask out pixels with large errors in Tk
    nh3_temp_data[eTk_data > eTk_lim] = np.NaN
    nh3_temp_data[nh3_temp_data == 0] = np.NaN
    maxTkin = plot_param['nh3_temp_lim'][1]
    finite_index = np.where(np.logical_and(np.isfinite(nh3_temp_data),nh3_temp_data < maxTkin))
    temp_ratio = h2_temp_data/nh3_temp_data
    fig = plt.figure()
    ax = plt.gca()
    plt.hexbin(np.log10(h2_col_data[finite_index]),temp_ratio[finite_index],
               gridsize=plot_param['temp_grid_size']*0.8,bins='log',cmap=plt.cm.pink_r,
               extent=[plot_param['h2_col_lim'][0],plot_param['h2_col_lim'][1],
                       0.5,3])#plot_param['her_temp_lim'][1]/plot_param['nh3_temp_lim'][1]])
    #plt.scatter(np.log10(h2_col_data[finite_index]),temp_ratio[finite_index])
    plt.figtext(0.72,0.8,'{0}'.format(region),fontsize=14,horizontalalignment='right')
    plt.plot(plot_param['h2_col_lim'],[1,1],linewidth=3,color='gray',alpha=0.6,label='$T_d = T_K$')
    plt.legend(loc=1,frameon=False)
    ax.set_ylabel(r'$T_\mathrm{d} \ / \ T_\mathrm{k}$')
    ax.set_xlabel(r'$N(\mathrm{H}_2) \ \mathrm{cm}^{-2}$')
    cb = plt.colorbar()
    cb.set_label('log10(N)')
    fig.savefig('figures/{0}_temp_ratio.pdf'.format(region))
    plt.close('all')

def plot_temp_h2_panel(nh3_tk_file,h2_td_file,h2_col_file,
                       eTk_file,eTk_lim,eTex_file,eTex_lim,
                       region,plot_param,ax):
    eTk_data  = fits.getdata(eTk_file)
    eTex_data = fits.getdata(eTex_file)
    nh3_temp_data = fits.getdata(nh3_tk_file)
    h2_temp_data  = fits.getdata(h2_td_file)
    h2_col_data   = fits.getdata(h2_col_file)
    # Mask out pixels with large errors in Tk
    nh3_temp_data[eTk_data > eTk_lim] = np.NaN
    nh3_temp_data[eTex_data > eTex_lim] = np.nan
    nh3_temp_data[nh3_temp_data == 0] = np.NaN
    maxTkin = plot_param['nh3_temp_lim'][1]
    finite_index = np.where(np.logical_and(np.isfinite(nh3_temp_data),nh3_temp_data < maxTkin))
    temp_ratio = h2_temp_data/nh3_temp_data
    im = ax.hexbin(np.log10(h2_col_data[finite_index]),temp_ratio[finite_index],
                   gridsize=plot_param['temp_grid_size']*0.8,bins='log',cmap=plt.cm.pink_r,
                   #extent=[plot_param['h2_col_lim'][0],plot_param['h2_col_lim'][1],
                   extent=[21.4,23.2,0.5,3])
                   #plot_param['her_temp_lim'][1]/plot_param['nh3_temp_lim'][1]])
    #plt.scatter(np.log10(h2_col_data[finite_index]),temp_ratio[finite_index])
    ax.text(0.93,0.72,'{0}'.format(region),fontsize=10,horizontalalignment='right',transform=ax.transAxes)
    ax.plot([21,24],[1,1],linewidth=2,color='gray',alpha=0.6,label='$T_d = T_K$')
    ax.legend(loc=1,frameon=False,fontsize=10)
    ax.set_ylabel(r'$T_\mathrm{d} \ / \ T_\mathrm{k}$',fontsize=12)
    ax.set_xlabel(r'$N(\mathrm{H}_2) \ \mathrm{cm}^{-2}$',fontsize=12)
    ax.tick_params(axis='both',which='major',labelsize=11)
    ax.set_xticks([21.5,22,22.5,23])
    ax.set_xlim(21.4,23.2)
    cb = fig.colorbar(im,ax=ax,ticks=[0,0.5,1])
    cb.set_label('log10(N)',fontsize=11)
    cb.ax.tick_params(labelsize=10)

def plot_temp_vs_td_panel(nh3_tk_file,h2_td_file,eTk_file,eTk_lim,eTex_file,eTex_lim,
                          region,plot_param,ax):
    eTk_data  = fits.getdata(eTk_file)
    eTex_data = fits.getdata(eTex_file)
    nh3_temp_data = fits.getdata(nh3_tk_file)
    h2_temp_data  = fits.getdata(h2_td_file)
    # Mask out pixels with large errors in Tk
    nh3_temp_data[eTk_data > eTk_lim] = np.NaN
    nh3_temp_data[eTex_data > eTex_lim] = np.nan
    nh3_temp_data[nh3_temp_data == 0] = np.NaN
    maxTkin = plot_param['nh3_temp_lim'][1]
    finite_index = np.where(np.logical_and(np.isfinite(nh3_temp_data),nh3_temp_data < maxTkin))
    temp_ratio = h2_temp_data/nh3_temp_data
    im = ax.hexbin(h2_temp_data[finite_index],temp_ratio[finite_index],
                   gridsize=plot_param['temp_grid_size']*0.8,bins='log',cmap=plt.cm.pink_r,
                   extent=[8,60,0.5,3])
    ax.text(0.07,0.74,'{0}'.format(region),fontsize=10,horizontalalignment='left',transform=ax.transAxes)
    ax.plot([0,100],[1,1],linewidth=2,color='gray',alpha=0.6,label='$T_d = T_K$')
    ax.legend(loc=2,frameon=False,fontsize=10)
    ax.set_ylabel(r'$T_\mathrm{d} \ / \ T_\mathrm{k}$',fontsize=12)
    ax.set_xlabel(r'$T_\mathrm{d}$ (K)',fontsize=12)
    ax.tick_params(axis='both',which='major',labelsize=11)
    ax.set_xlim(8,60)
    cb = fig.colorbar(im,ax=ax,ticks=[0,0.5,1,1.5,2])
    cb.set_label('log10(N)',fontsize=11)
    cb.ax.tick_params(labelsize=10)

def plot_tk_vs_td_panel(nh3_tk_file,h2_td_file,eTk_file,eTk_lim,eTex_file,eTex_lim,
                        region,plot_param,ax):
    eTk_data  = fits.getdata(eTk_file)
    eTex_data = fits.getdata(eTex_file)
    nh3_temp_data = fits.getdata(nh3_tk_file)
    h2_temp_data  = fits.getdata(h2_td_file)
    # Mask out pixels with large errors in Tk
    nh3_temp_data[eTk_data > eTk_lim] = np.NaN
    nh3_temp_data[eTex_data > eTex_lim] = np.nan
    nh3_temp_data[nh3_temp_data == 0] = np.NaN
    maxTkin = plot_param['nh3_temp_lim'][1]
    finite_index = np.where(np.logical_and(np.isfinite(nh3_temp_data),nh3_temp_data < maxTkin))
    temp_ratio = h2_temp_data/nh3_temp_data
    im = ax.hexbin(nh3_temp_data[finite_index], h2_temp_data[finite_index],
                   gridsize=plot_param['temp_grid_size'],bins='log',cmap=plt.cm.pink_r,
                   extent=[5,40,5,60])
    ax.text(0.07,0.85,'{0}'.format(region),fontsize=10,horizontalalignment='left',transform=ax.transAxes)
    ax.plot([0,100],[0,100],linewidth=2,color='gray',alpha=0.6,label='$T_d = T_K$')
    ax.legend(loc=4,frameon=False,fontsize=10)
    ax.set_xlabel(r'$T_\mathrm{k}$ (K)',fontsize=12)
    ax.set_ylabel(r'$T_\mathrm{d}$ (K)',fontsize=12)
    ax.tick_params(axis='both',which='major',labelsize=11)
    ax.set_ylim(5,60)
    ax.set_xlim(5,40)
    cb = fig.colorbar(im,ax=ax,ticks=[0,0.5,1,1.5,2])
    cb.set_label('log10(N)',fontsize=11)
    cb.ax.tick_params(labelsize=10)

def plot_td_h2_panel(nh3_tk_file,h2_td_file,h2_col_file,
                     eTk_file,eTk_lim,eTex_file,eTex_lim,
                     region,plot_param,ax):
    eTk_data  = fits.getdata(eTk_file)
    eTex_data = fits.getdata(eTex_file)
    nh3_temp_data = fits.getdata(nh3_tk_file)
    h2_temp_data  = fits.getdata(h2_td_file)
    h2_col_data   = fits.getdata(h2_col_file)
    # Mask out pixels with large errors in Tk
    nh3_temp_data[eTk_data > eTk_lim] = np.NaN
    nh3_temp_data[eTex_data > eTex_lim] = np.nan
    nh3_temp_data[nh3_temp_data == 0] = np.NaN
    maxTkin = plot_param['nh3_temp_lim'][1]
    finite_index = np.where(np.logical_and(np.isfinite(nh3_temp_data),nh3_temp_data < maxTkin))
    td_mean = np.nanmedian(h2_temp_data[finite_index])
    tk_mean = np.nanmedian(nh3_temp_data[finite_index])
    im = ax.hexbin(np.log10(h2_col_data[finite_index]),h2_temp_data[finite_index],
                   gridsize=plot_param['temp_grid_size'],bins='log',cmap=plt.cm.pink_r,
                   extent=[21.4,23.2,5,60])
    #plt.scatter(np.log10(h2_col_data[finite_index]),temp_ratio[finite_index])
    ax.text(0.93,0.72,'{0}'.format(region),fontsize=10,horizontalalignment='right',transform=ax.transAxes)
    ax.plot([20,24],[td_mean,td_mean],color='red',alpha=0.3,linewidth=1.5,label=r'$\langle T_d \rangle$')
    ax.plot([20,24],[tk_mean,tk_mean],color='blue',alpha=0.3,linewidth=1.5,label=r'$\langle T_\mathrm{K} \rangle$')
    #ax.legend(loc=2,frameon=False,fontsize=10)
    ax.set_ylabel(r'$T_\mathrm{d}$ (K)',fontsize=12)
    ax.set_xlabel(r'$N(\mathrm{H}_2) \ \mathrm{cm}^{-2}$',fontsize=12)
    ax.tick_params(axis='both',which='major',labelsize=11)
    ax.set_xticks([21.5,22,22.5,23])
    ax.set_xlim(21.4,23.2)
    cb = fig.colorbar(im,ax=ax,ticks=[0,0.5,1])
    cb.set_label('log10(N)',fontsize=11)
    cb.ax.tick_params(labelsize=10)

def plot_tk_h2_panel(nh3_tk_file,h2_td_file,h2_col_file,
                     eTk_file,eTk_lim,eTex_file,eTex_lim,
                     region,plot_param,ax):
    eTk_data  = fits.getdata(eTk_file)
    eTex_data = fits.getdata(eTex_file)
    nh3_temp_data = fits.getdata(nh3_tk_file)
    h2_temp_data  = fits.getdata(h2_td_file)
    h2_col_data   = fits.getdata(h2_col_file)
    # Mask out pixels with large errors in Tk
    nh3_temp_data[eTk_data > eTk_lim] = np.NaN
    nh3_temp_data[eTex_data > eTex_lim] = np.nan
    nh3_temp_data[nh3_temp_data == 0] = np.NaN
    maxTkin = plot_param['nh3_temp_lim'][1]
    finite_index = np.where(np.logical_and(np.isfinite(nh3_temp_data),nh3_temp_data < maxTkin))
    td_mean = np.nanmedian(h2_temp_data[finite_index])
    tk_mean = np.nanmedian(nh3_temp_data[finite_index])
    im = ax.hexbin(np.log10(h2_col_data[finite_index]),nh3_temp_data[finite_index],
                   gridsize=plot_param['temp_grid_size'],bins='log',cmap=plt.cm.pink_r,
                   extent=[21.4,23.2,5,60])
    #plt.scatter(np.log10(h2_col_data[finite_index]),temp_ratio[finite_index])
    ax.plot([20,24],[td_mean,td_mean],color='red',alpha=0.3,linewidth=1.5,label=r'$\langle T_d \rangle$')
    ax.plot([20,24],[tk_mean,tk_mean],color='blue',alpha=0.3,linewidth=1.5,label=r'$\langle T_\mathrm{K} \rangle$')
    ax.text(0.93,0.72,'{0}'.format(region),fontsize=10,horizontalalignment='right',transform=ax.transAxes)
    ax.legend(loc=2,frameon=False,fontsize=10)
    ax.set_ylabel(r'$T_\mathrm{K}$ (K)',fontsize=12)
    ax.set_xlabel(r'$N(\mathrm{H}_2) \ \mathrm{cm}^{-2}$',fontsize=12)
    ax.tick_params(axis='both',which='major',labelsize=11)
    ax.set_xticks([21.5,22,22.5,23])
    ax.set_xlim(21.4,23.2)
    cb = fig.colorbar(im,ax=ax,ticks=[0,0.5,1])
    cb.set_label('log10(N)',fontsize=11)
    cb.ax.tick_params(labelsize=10)

###
herschel_nh2_dir = 'nh2_regridded'
herschel_td_dir  = 'Td_regridded'
herschel_im_dir  = 'h500_regridded'
region_list  = ['B18','NGC1333','L1688','OrionA']
file_extension = 'DR1_rebase3'
# Stick with DR1 or use more extended regions? L1688 is bigger, for example. 
# Set uncertainty limit on Tex for good N(NH3) fits
# Need to mask out Orion nebula - use sky regions ** do this for X(NH3) analysis too. 
eTex_lim = 1. # K
eTk_lim = 2. # K
snr_lim = 4.

for region in region_list:
    plot_param = plottingDictionary[region]
    if region in ['OrionA']:
        herschelTdFile = '{0}/{1}_Td_regrid_masked.fits'.format(herschel_td_dir,region)
        herschelNH2File = '{0}/{1}_NH2_regrid_masked.fits'.format(herschel_nh2_dir,region)
    else:
        herschelTdFile = '{0}/{1}_Td_regrid.fits'.format(herschel_td_dir,region)
        herschelNH2File = '{0}/{1}_NH2_regrid.fits'.format(herschel_nh2_dir,region)
    #herschelImFile = '{0}/{1}_500_regrid.fits'.format(herschel_im_dir,region)
    protList = 'protostars/{0}_protostar_list.txt'.format(region)
    # Here, care about SNR in (2,2) line
    nh3ImFits = 'nh3_data/{0}_NH3_22_{1}_Tpeak.fits'.format(region,file_extension)
    nh3RmsFits = 'nh3_data/{0}_NH3_22_{1}_rms.fits'.format(region,file_extension)
    nh3TkFits  = 'propertyMaps/{0}_Tkin_{1}_flag.fits'.format(region,file_extension)
    nh3eTkFits = 'propertyMaps/{0}_eTkin_{1}_flag.fits'.format(region,file_extension)
    #nh3ColFits = 'propertyMaps/{0}_N_NH3_{1}_flag.fits'.format(region,file_extension)
    nh3eTexFits = 'propertyMaps/{0}_eTex_{1}_flag.fits'.format(region,file_extension)
    # Get SNR in (2,2) line
    tpeak_22 = fits.getdata(nh3ImFits)
    rms_22   = fits.getdata(nh3RmsFits)
    snr_22   = tpeak_22/rms_22
    # Look at dust and gas temperatures
    plot_temp_compare(nh3TkFits,herschelTdFile,nh3eTkFits,eTk_lim,region,plot_param)
    #plot_temp_compare(nh3TkFits,herschelTdFile,snr_22,snr_lim,region,plot_param)
    # Compare Tk, Td as a function of N(H2)
    plot_temp_h2(nh3TkFits,herschelTdFile,herschelNH2File,nh3eTkFits,eTk_lim,region,plot_param)
    # Figure showing ratio of Tk/Td over regions
    plot_temp_overlay(nh3TkFits,herschelTdFile,nh3ImFits,region,plot_param,include_prot=True)
    

# Set up for panel plot
fig,axes = plt.subplots(2,2,figsize=(7.5,5))
axes = axes.ravel()

for i in range(len(region_list)):
    region = region_list[i]
    plot_param = plottingDictionary[region]
    if region in ['OrionA']:
        herschelTdFile = '{0}/{1}_Td_regrid_masked.fits'.format(herschel_td_dir,region)
        herschelNH2File = '{0}/{1}_NH2_regrid_masked.fits'.format(herschel_nh2_dir,region)
    else:
        herschelTdFile = '{0}/{1}_Td_regrid.fits'.format(herschel_td_dir,region)
        herschelNH2File = '{0}/{1}_NH2_regrid.fits'.format(herschel_nh2_dir,region)
    protList = 'protostars/{0}_protostar_list.txt'.format(region)
    nh3TkFits  = 'propertyMaps/{0}_Tkin_{1}_flag.fits'.format(region,file_extension)
    nh3eTkFits = 'propertyMaps/{0}_eTkin_{1}_flag.fits'.format(region,file_extension)
    nh3eTexFits = 'propertyMaps/{0}_eTex_{1}_flag.fits'.format(region,file_extension)
    # And plot
    ax = axes[i]
    plot_temp_h2_panel(nh3TkFits,herschelTdFile,herschelNH2File,
                       nh3eTkFits,eTk_lim,nh3eTexFits,eTex_lim,
                       region,plot_param,ax)

plt.tight_layout()
fig.savefig('figures/DR1_temp_ratio.pdf')
plt.close('all')

# Set up for panel plot
fig,axes = plt.subplots(2,2,figsize=(7.5,5))
axes = axes.ravel()

for i in range(len(region_list)):
    region = region_list[i]
    plot_param = plottingDictionary[region]
    if region in ['OrionA']:
        herschelTdFile = '{0}/{1}_Td_regrid_masked.fits'.format(herschel_td_dir,region)
        herschelNH2File = '{0}/{1}_NH2_regrid_masked.fits'.format(herschel_nh2_dir,region)
    else:
        herschelTdFile = '{0}/{1}_Td_regrid.fits'.format(herschel_td_dir,region)
        herschelNH2File = '{0}/{1}_NH2_regrid.fits'.format(herschel_nh2_dir,region)
    protList = 'protostars/{0}_protostar_list.txt'.format(region)
    nh3TkFits  = 'propertyMaps/{0}_Tkin_{1}_flag.fits'.format(region,file_extension)
    nh3eTkFits = 'propertyMaps/{0}_eTkin_{1}_flag.fits'.format(region,file_extension)
    nh3eTexFits = 'propertyMaps/{0}_eTex_{1}_flag.fits'.format(region,file_extension)
    # And plot
    ax = axes[i]
    plot_temp_vs_td_panel(nh3TkFits,herschelTdFile,nh3eTkFits,eTk_lim,nh3eTexFits,eTex_lim,
                          region,plot_param,ax)

plt.tight_layout()
fig.savefig('figures/DR1_temp_ratio_vs_td.pdf')
plt.close('all')

# Set up for panel plot
fig,axes = plt.subplots(2,2,figsize=(7.5,5))
axes = axes.ravel()

for i in range(len(region_list)):
    region = region_list[i]
    plot_param = plottingDictionary[region]
    if region in ['OrionA']:
        herschelTdFile = '{0}/{1}_Td_regrid_masked.fits'.format(herschel_td_dir,region)
        herschelNH2File = '{0}/{1}_NH2_regrid_masked.fits'.format(herschel_nh2_dir,region)
    else:
        herschelTdFile = '{0}/{1}_Td_regrid.fits'.format(herschel_td_dir,region)
        herschelNH2File = '{0}/{1}_NH2_regrid.fits'.format(herschel_nh2_dir,region)
    protList = 'protostars/{0}_protostar_list.txt'.format(region)
    nh3TkFits  = 'propertyMaps/{0}_Tkin_{1}_flag.fits'.format(region,file_extension)
    nh3eTkFits = 'propertyMaps/{0}_eTkin_{1}_flag.fits'.format(region,file_extension)
    nh3eTexFits = 'propertyMaps/{0}_eTex_{1}_flag.fits'.format(region,file_extension)
    # And plot
    ax = axes[i]
    plot_tk_vs_td_panel(nh3TkFits,herschelTdFile,nh3eTkFits,eTk_lim,nh3eTexFits,eTex_lim,
                        region,plot_param,ax)

plt.tight_layout()
fig.savefig('figures/DR1_tk_vs_td.pdf')
plt.close('all')

# Set up for panel plot
fig,axes = plt.subplots(4,2,figsize=(7.5,10))
axes = axes.ravel()

for i in range(len(region_list)):
    region = region_list[i]
    plot_param = plottingDictionary[region]
    if region in ['OrionA']:
        herschelTdFile = '{0}/{1}_Td_regrid_masked.fits'.format(herschel_td_dir,region)
        herschelNH2File = '{0}/{1}_NH2_regrid_masked.fits'.format(herschel_nh2_dir,region)
    else:
        herschelTdFile = '{0}/{1}_Td_regrid.fits'.format(herschel_td_dir,region)
        herschelNH2File = '{0}/{1}_NH2_regrid.fits'.format(herschel_nh2_dir,region)
    protList = 'protostars/{0}_protostar_list.txt'.format(region)
    nh3TkFits  = 'propertyMaps/{0}_Tkin_{1}_flag.fits'.format(region,file_extension)
    nh3eTkFits = 'propertyMaps/{0}_eTkin_{1}_flag.fits'.format(region,file_extension)
    nh3eTexFits = 'propertyMaps/{0}_eTex_{1}_flag.fits'.format(region,file_extension)
    # And plot
    ax = axes[2*i]
    plot_tk_h2_panel(nh3TkFits,herschelTdFile,herschelNH2File,
                     nh3eTkFits,eTk_lim,nh3eTexFits,eTex_lim,
                     region,plot_param,ax)
    ax = axes[2*i+1]
    plot_td_h2_panel(nh3TkFits,herschelTdFile,herschelNH2File,
                     nh3eTkFits,eTk_lim,nh3eTexFits,eTex_lim,
                     region,plot_param,ax)   

plt.tight_layout()
fig.savefig('figures/DR1_temps_vs_h2.pdf')
plt.close('all')
