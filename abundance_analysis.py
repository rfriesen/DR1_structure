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

def get_prot_loc(region):
    protFile = 'protostars/{0}_protostar_list.txt'.format(region)
    ra, dec = np.loadtxt(protFile,unpack=True)
    return ra, dec

def log10_h2(h2_image):
    log_h2_data = np.log10(h2_image.data)
    log_h2_hdu = fits.PrimaryHDU(log_h2_data,h2_image.header)
    return log_h2_hdu    

def plot_abundance(nh3_cont_fits,nh3_col_hdu,h2_col_hdu,region,plot_pars,maskLim,obsMaskFits):
    text_size = 14
    b18_text_size = 20
    if region == 'B18':
        text_size = b18_text_size
    # Get protostellar locations
    ra_prot, de_prot = get_prot_loc(region)
    # Contour parameters (currently NH3 moment 0)
    cont_color='0.6'
    cont_lw   = 0.6
    cont_levs=2**np.arange( 0,20)*plot_param['w11_step']
    # Calculate abundance
    log_xnh3 = nh3_col_hdu[0].data - np.log10(h2_col_hdu.data)
    log_xnh3_hdu = fits.PrimaryHDU(log_xnh3,nh3_col_hdu[0].header)
    log_xnh3_hdu.writeto('../testing/{0}/parameterMaps/{0}_XNH3_{1}.fits'.format(region,file_extension),clobber=True)
    fig=aplpy.FITSFigure(log_xnh3_hdu,figsize=(plot_param['size_x'], plot_param['size_y']))
    fig.show_colorscale(cmap='YlOrRd_r',vmin=plot_param['xnh3_lim'][0],vmax=plot_param['xnh3_lim'][1])
    #fig.set_nan_color('0.95')
    # Observations mask contour
    fig.show_contour(obsMaskFits,colors='white',levels=1,linewidths=1.5)
    # NH3 moment contours
    # Masking of small (noisy) regions
    selem = np.array([[0,1,0],[1,1,1],[0,1,0]])
    LowestContour = cont_levs[0]*0.5
    w11_hdu = fits.open(nh3_cont_fits)
    map = w11_hdu[0].data
    mask = binary_opening(map > LowestContour, selem)
    MaskedMap = mask*map
    w11_hdu[0].data = MaskedMap
    fig.show_contour(w11_hdu,colors=cont_color,levels=cont_levs,linewidths=cont_lw)
    # Ticks
    fig.ticks.set_color('black')
    fig.tick_labels.set_font(family='sans_serif',size=text_size)
    fig.tick_labels.set_xformat('hh:mm:ss')
    fig.tick_labels.set_style('colons')
    fig.tick_labels.set_yformat('dd:mm')
    # Scale bar
    ang_sep = (plot_param['scalebar_size'].to(u.au)/plot_param['distance']).to(u.arcsec, equivalencies=u.dimensionless_angles())
    fig.add_colorbar()
    fig.colorbar.show(box_orientation='horizontal', width=0.1, pad=0.0, location='top',
                      ticks=[-10,-9.5,-9,-8.5,-8,-7.5,-7,-6.5])
    fig.colorbar.set_font(family='sans_serif',size=text_size)
    fig.add_scalebar(ang_sep.to(u.degree))
    fig.scalebar.set_font(family='sans_serif',size=text_size)
    fig.scalebar.set_corner(plot_param['scalebar_pos'])
    fig.scalebar.set(color='black')
    fig.scalebar.set_label('{0:4.2f}'.format(plot_param['scalebar_size']))
    label_colour = 'black'
    fig.add_label(plot_param['label_xpos'], plot_param['label_ypos'], 
                  '{0}\n{1}'.format(region,r'$\mathrm{log} \ X(\mathrm{NH}_3)$'), 
                  relative=True, color=label_colour, 
                  horizontalalignment=plot_param['label_align'],
                  family='sans_serif',size=text_size)
    fig.save( 'figures/{0}_xnh3_image.pdf'.format(region),adjust_bbox=True,dpi=200)#, bbox_inches='tight')
    # Add protostars
    fig.show_markers(ra_prot,de_prot,marker='*',s=50,
                     c='white',edgecolors='black',linewidth=0.5,zorder=4)
    fig.save( 'figures/{0}_xnh3_image_prot.pdf'.format(region),adjust_bbox=True,dpi=200)
    fig.close()

def plot_column_compare(nh3_col_file,h2_col_file,nh3_etex_file,eTex_lim,region,plot_param):
    nh3_col_data = fits.getdata(nh3_col_file)
    h2_col_data = fits.getdata(h2_col_file)
    # Mask where no good fits (data == 0)
    nh3_col_data[(nh3_col_data == 0)] = np.nan
    # Mask where uncertainties on Tex are high
    nh3eTex = fits.getdata(nh3_etex_file)
    nh3_col_data[(nh3eTex > eTex_lim)] = np.nan
    finite_index = np.where(np.isfinite(nh3_col_data))

    gridsize = plot_param['nh2_grid_size']
    # Plots
    fig = plt.figure(figsize=(5,3.5))
    ax = plt.gca()
    # NH3 column already in log
    # Use hexbin, scatter plots not useful here
    plt.hexbin(np.log10(h2_col_data[finite_index]),
               nh3_col_data[finite_index],
               gridsize=gridsize,cmap=plt.cm.pink_r,bins='log',
               extent=[plot_param['h2_col_lim'][0],plot_param['h2_col_lim'][1],
                       plot_param['nh3_col_lim'][0],plot_param['nh3_col_lim'][1]])
    ax.set_ylabel('log $N$(para-NH$_3$) (cm$^{-2}$)')
    ax.set_xlabel('log $N$(H$_2$) (cm$^{-2}$')
    plt.figtext(0.6,0.25,'{0}'.format(region),fontsize=14)
    # Add lines for typical constant abundances. Fixing points messes up x,y plot limits.
    if region == 'OrionA':
        ax.plot(np.array([21.3,23.6]),np.array([21.3,23.6])-8,
                linewidth=3,color='gray',alpha=0.6,zorder=4,
                label=r'$X(\mathrm{NH}_3) = 10^{-8}$')
        #ax.plot(np.array([21.7,24.6])-9,np.array([21.7,24.6]),
        #        linewidth=3,color='gray',linestyle='--',alpha=0.6,zorder=4,
        #        label=r'$X(\mathrm{NH}_3) = 10^{-9}$')
    else:
        ax.plot(plot_param['h2_col_lim'],plot_param['h2_col_lim']-8,
                linewidth=3,color='gray',alpha=0.6,zorder=4,
                label=r'$X(\mathrm{NH}_3) = 10^{-8}$')
        #ax.plot(plot_param['h2_col_lim']-9,plot_param['h2_col_lim'],
        #        linewidth=3,color='gray',linestyle='--',alpha=0.6,zorder=4,
        #        label=r'$X(\mathrm{NH}_3) = 10^{-9}$')
    #ax.set_ylim(0,herMax[region_i])
    cb = plt.colorbar()
    cb.set_label('log10(N)')
    plt.legend(loc=2,frameon=False)
    plt.tight_layout()
    fig.savefig('figures/{0}_Herschel_vs_NH3_col.pdf'.format(region))
    plt.close('all')

def plot_abundance_compare(nh3_col_file,h2_col_file,nh3_etex_file,eTex_lim,region,plot_param):
    h2_col_data = fits.getdata(h2_col_file)
    nh3_col_data = fits.getdata(nh3_col_file)

    # Mask where no good fits (data == 0)
    nh3_col_data[(nh3_col_data == 0)] = np.nan
    # Mask where uncertainties on Tex are high
    nh3eTex = fits.getdata(nh3_etex_file)
    nh3_col_data[(nh3eTex > eTex_lim)] = np.nan
    # Mask where no good fits (data == 0)
    nh3_col_data[(nh3_col_data == 0)] = np.nan

    # N(NH3) data in logspace, N(H2) data not
    xnh3_data = nh3_col_data - np.log10(h2_col_data)

    gridsize = plot_param['nh2_grid_size']
    finite_index = np.where(np.isfinite(nh3_col_data))
    fig = plt.figure(figsize=(5,3.5))
    ax = plt.gca()
    # NH3 abundance already in log
    # Use hexbin, scatter plots not useful here
    plt.hexbin(np.log10(h2_col_data[finite_index]),
               xnh3_data[finite_index],
               gridsize=gridsize,cmap=plt.cm.pink_r,bins='log',
               extent=[plot_param['h2_col_lim'][0],plot_param['h2_col_lim'][1],
                       plot_param['xnh3_lim'][0],plot_param['xnh3_lim'][1]])
    ax.set_ylabel('log $X$(para-NH$_3$) (cm$^{-2}$)')
    ax.set_xlabel('log $N$(H$_2$) (cm$^{-2}$')
    plt.figtext(0.25,0.25,'{0}'.format(region),fontsize=14)
    cb = plt.colorbar()
    cb.set_label('log10(N)')
    plt.legend(loc=2,frameon=False)
    plt.tight_layout()
    fig.savefig('figures/{0}_Herschel_vs_XNH3.pdf'.format(region),)
    plt.close('all')


def plot_flux_compare(nh3ImFile,h2ImFile,region,plot_param):
    max_nh3 = plot_param['nh3_flux_max']
    max_her = plot_param['her_flux_max']
    gridsize = plot_param['flux_grid_size']
    nh3_data = fits.getdata(nh3ImFile)
    h2_data  = fits.getdata(h2ImFile)
    finite_index = np.where(np.isfinite(nh3_data))
    fig = plt.figure()
    ax = plt.gca()
    plt.hexbin(nh3_data[finite_index],h2_data[finite_index],
               gridsize=gridsize,bins='log',cmap=plt.cm.YlOrRd)
    ax.set_xlabel('$\int T dv$ (K km s$^{-1}$)')
    ax.set_ylabel('500 $\mu$m (MJy sr$^{-1}$)')
    ax.set_xlim(-1,max_nh3)
    ax.set_ylim(0,max_her)
    cb = plt.colorbar()
    cb.set_label('log10(N)')
    fig.savefig('figures/{0}_Herschel_vs_NH3_flux.pdf'.format(region))    

def plot_co_compare(xImFile,coImFile,region,plot_param):
    gridsize=plot_param['nh2_grid_size']
    xnh3_data = fits.getdata(xImFile)
    co_data   = fits.getdata(coImFile)
    finite_index = np.where(np.isfinite(xnh3_data))
    fig = plt.figure()
    ax = plt.gca()
    plt.hexbin(xnh3_data[finite_index],co_data[finite_index],
               gridsize=gridsize,bins='log',cmap=plt.cm.YlOrRd)
    ax.set_xlabel('X(NH$_3$)')
    ax.set_ylabel('CO mom 0 (K km s$^{-1}$)')
    ax.set_xlim(plot_param['xnh3_lim'])
    #ax.set_ylim(0,max_her)
    cb = plt.colorbar()
    cb.set_label('log10(N)')
    fig.savefig('figures/{0}_XNH3_vs_CO.pdf'.format(region))    

def plot_tex_compare(xImFile,nh3_tex_file,nh3_etex_file,eTex_lim,region,plot_param):
    gridsize=plot_param['temp_grid_size']
    xnh3_data = fits.getdata(xImFile)
    tex_data  = fits.getdata(nh3_tex_file)
    # Mask where uncertainties on Tex are high
    nh3eTex = fits.getdata(nh3_etex_file)
    xnh3_data[(nh3eTex > eTex_lim)] = np.nan
    finite_index = np.where(np.isfinite(xnh3_data))
    fig = plt.figure()
    ax = plt.gca()
    plt.hexbin(tex_data[finite_index],xnh3_data[finite_index],
               gridsize=gridsize,bins='log',cmap=plt.cm.pink_r)
    ax.set_ylabel('X(NH$_3$)')
    ax.set_xlabel('T$_{ex}$ (K)')
    ax.set_ylim(plot_param['xnh3_lim'])
    #ax.set_ylim(0,max_her)
    cb = plt.colorbar()
    cb.set_label('log10(N)')
    fig.savefig('figures/{0}_XNH3_vs_Tex.pdf'.format(region))    

def plot_abundance_compare_panel(nh3_col_file,h2_col_file,
                                 nh3_tex_file, tex_lim, nh3_etex_file,eTex_lim,
                                 region,plot_param,ax):
    h2_col_data = fits.getdata(h2_col_file)
    nh3_col_data = fits.getdata(nh3_col_file)
    # Mask where no good fits (data == 0)
    nh3_col_data[(nh3_col_data == 0)] = np.nan
    # Mask where uncertainties on Tex are high
    nh3Tex  = fits.getdata(nh3_tex_file)
    nh3eTex = fits.getdata(nh3_etex_file)
    nh3_col_data[nh3Tex < tex_lim] = np.nan
    nh3_col_data[(nh3eTex > eTex_lim)] = np.nan
    # Mask where no good fits (data == 0)
    nh3_col_data[(nh3_col_data == 0)] = np.nan
    # N(NH3) data in logspace, N(H2) data not
    xnh3_data = nh3_col_data - np.log10(h2_col_data)

    gridsize = plot_param['nh2_grid_size']
    finite_index = np.where(np.isfinite(nh3_col_data))
    # Get mean, median X(NH3) value:
    medX = np.nanmedian(xnh3_data[finite_index])
    meanX = np.nanmean(xnh3_data[finite_index])
    # NH3 abundance already in log
    # Use hexbin, scatter plots not useful here
    im = ax.hexbin(np.log10(h2_col_data[finite_index]),
                   xnh3_data[finite_index],
                   gridsize=gridsize,cmap=plt.cm.pink_r,bins='log',
                   extent=[plot_param['h2_col_lim'][0],plot_param['h2_col_lim'][1],
                           plot_param['xnh3_lim'][0],plot_param['xnh3_lim'][1]])
    #ax.plot([21,24],[medX,medX],linewidth=2,color='gray',alpha=0.6,
    #        label='$X$(NH$_3$) = {:3.1f}'.format(medX))
    ax.set_ylabel('log $X$(para-NH$_3$)',fontsize=12)
    ax.set_xlabel('log $N$(H$_2$) (cm$^{-2}$)',fontsize=12)
    ax.set_xlim(21.1,23.2)
    ax.set_ylim(-9,-7)
    ax.set_yticks([-9,-8,-7])
    ax.tick_params(axis='both',which='major',labelsize=11)
    ax.text(0.93,0.72,'{0}'.format(region),fontsize=10,horizontalalignment='right',transform=ax.transAxes)
    minor_locator = AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.yaxis.set_minor_locator(minor_locator)
    cb = fig.colorbar(im,ax=ax,ticks=[0,0.5,1])
    cb.set_label('log10(N)',fontsize=11)
    cb.ax.tick_params(labelsize=10)
    ax.legend(loc=1,frameon=False,fontsize=10)

def plot_abundance_compare_panel_22(nh3_col_file,h2_col_file,
                                    nh3_tex_file, tex_lim, nh3_etex_file,eTex_lim,
                                    nh3_22_peak, nh3_22_rms,
                                    region,plot_param,ax):
    h2_col_data = fits.getdata(h2_col_file)
    nh3_col_data = fits.getdata(nh3_col_file)
    nh3_22_data = fits.getdata(nh3_22_peak)
    nh3_22_rms_data = fits.getdata(nh3_22_rms)
    snr_22 = nh3_22_data/nh3_22_rms_data
    # Mask where snr_22 < 5
    nh3_col_data[snr_22 < plot_param['snr_lim']] = np.nan
    #nh3_col_data[snr_22 < 2.5] = np.nan
    # Mask where no good fits (data == 0)
    nh3_col_data[(nh3_col_data == 0)] = np.nan
    # Mask where uncertainties on Tex are high
    nh3Tex  = fits.getdata(nh3_tex_file)
    nh3eTex = fits.getdata(nh3_etex_file)
    #nh3_col_data[nh3Tex < tex_lim] = np.nan
    nh3_col_data[(nh3eTex > eTex_lim)] = np.nan
    # Mask where no good fits (data == 0)
    nh3_col_data[(nh3_col_data == 0)] = np.nan
    # N(NH3) data in logspace, N(H2) data not
    xnh3_data = nh3_col_data - np.log10(h2_col_data)

    gridsize = plot_param['nh2_grid_size']
    finite_index = np.where(np.isfinite(nh3_col_data))
    # Get mean, median X(NH3) value:
    medX = np.nanmedian(xnh3_data[finite_index])
    meanX = np.nanmean(xnh3_data[finite_index])
    # NH3 abundance already in log
    # Use hexbin, scatter plots not useful here
    im = ax.hexbin(np.log10(h2_col_data[finite_index]),
                   xnh3_data[finite_index],
                   gridsize=gridsize,cmap=plt.cm.pink_r,bins='log',
                   extent=[plot_param['h2_col_lim'][0],plot_param['h2_col_lim'][1],
                           plot_param['xnh3_lim'][0],plot_param['xnh3_lim'][1]])
    #ax.plot([21,24],[medX,medX],linewidth=2,color='gray',alpha=0.6,
    #        label='$X$(NH$_3$) = {:3.1f}'.format(medX))
    ax.set_ylabel('log $X$(para-NH$_3$)',fontsize=12)
    ax.set_xlabel('log $N$(H$_2$) (cm$^{-2}$)',fontsize=12)
    ax.set_xlim(21.1,23.2)
    ax.set_ylim(-9,-7)
    ax.set_yticks([-9,-8,-7])
    ax.tick_params(axis='both',which='major',labelsize=11)
    ax.text(0.93,0.72,'{0}'.format(region),fontsize=10,horizontalalignment='right',transform=ax.transAxes)
    minor_locator = AutoMinorLocator(2)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.yaxis.set_minor_locator(minor_locator)
    cb = fig.colorbar(im,ax=ax,ticks=[0,0.5,1])
    cb.set_label('log10(N)',fontsize=11)
    cb.ax.tick_params(labelsize=10)
    ax.legend(loc=1,frameon=False,fontsize=10)

###
herschel_nh2_dir = 'nh2_regridded'
herschel_td_dir  = 'Td_regridded'
herschel_im_dir  = 'h500_regridded'
region_list  = ['B18','NGC1333','L1688','OrionA']
file_extension = 'DR1_rebase3'
# Stick with DR1 or use more extended regions? L1688 is bigger, for example. 
# Set uncertainty limit on Tex for good N(NH3) fits
eTex_lim = 1. # K
Tex_lim  = 0.  # K
eTk_lim = 2. # K
# Maybe add in additional mask based on NH3 (2,2) data

for region in region_list:
    plot_param = plottingDictionary[region]
    if region in ['OrionA']:
        herschelNH2File = '{0}/{1}_NH2_regrid_masked.fits'.format(herschel_nh2_dir,region)
        herschelTdFile = '{0}/{1}_Td_regrid_masked.fits'.format(herschel_td_dir,region)
    else:
        herschelNH2File = '{0}/{1}_NH2_regrid.fits'.format(herschel_nh2_dir,region)
        herschelTdFile = '{0}/{1}_Td_regrid.fits'.format(herschel_td_dir,region)
    herschelImFile = '{0}/{1}_500_regrid.fits'.format(herschel_im_dir,region)
    protList = 'protostars/{0}_protostar_list.txt'.format(region)
    nh3ImFits = 'nh3_data/{0}_NH3_11_{1}_mom0_QA_trim.fits'.format(region,file_extension)
    nh3RmsFits = 'nh3_data/{0}_NH3_11_{1}_rms_QA_trim.fits'.format(region,file_extension)
    nh3TkFits  = 'propertyMaps/{0}_Tkin_{1}_flag.fits'.format(region,file_extension)
    nh3eTkFits = 'propertyMaps/{0}_eTkin_{1}_flag.fits'.format(region,file_extension)
    nh3ColFits = 'propertyMaps/{0}_N_NH3_{1}_flag.fits'.format(region,file_extension)
    nh3TexFits = 'propertyMaps/{0}_Tex_{1}_flag.fits'.format(region,file_extension)
    nh3eTexFits = 'propertyMaps/{0}_eTex_{1}_flag.fits'.format(region,file_extension)
    # First compare flux
    plot_flux_compare(nh3ImFits,herschelImFile,region,plot_param)
    # Next look at column densities and abundances
    plot_column_compare(nh3ColFits,herschelNH2File,nh3eTexFits,eTex_lim,region,plot_param)
    plot_abundance_compare(nh3ColFits,herschelNH2File,nh3eTexFits,eTex_lim,region,plot_param)
    xnh3File = 'propertyMaps/{0}_XNH3_{1}.fits'.format(region,file_extension)
    plot_tex_compare(xnh3File,nh3TexFits,nh3eTexFits,eTex_lim,region,plot_param)
    if region in ['NGC1333']:
        coFile = '../otherData/jcmt/{0}/match_NH3/ngc1333_c18o32_jcmt_reprojected_mom0.fits'.format(region)
        plot_co_compare(xnh3File,coFile,region,plot_param)

# Set up for panel plot
fig,axes = plt.subplots(2,2,figsize=(7.5,5))
axes = axes.ravel()

for i in range(len(region_list)):
    region = region_list[i]
    plot_param = plottingDictionary[region]
    if region in ['OrionA']:
        herschelNH2File = '{0}/{1}_NH2_regrid_masked.fits'.format(herschel_nh2_dir,region)
    else:
        herschelNH2File = '{0}/{1}_NH2_regrid.fits'.format(herschel_nh2_dir,region)
    nh322Fits = 'nh3_data/{0}_NH3_22_{1}_mom0_QA_trim.fits'.format(region,file_extension)
    nh322RmsFits = 'nh3_data/{0}_NH3_22_{1}_rms_QA_trim.fits'.format(region,file_extension)
    nh3ColFits = 'propertyMaps/{0}_N_NH3_{1}_flag.fits'.format(region,file_extension)
    nh3TexFits = 'propertyMaps/{0}_Tex_{1}_flag.fits'.format(region,file_extension)
    nh3eTexFits = 'propertyMaps/{0}_eTex_{1}_flag.fits'.format(region,file_extension)
    # And plot
    ax = axes[i]
    #plot_abundance_compare_panel(nh3ColFits,herschelNH2File,nh3TexFits,plot_param['tex_lim'],
    #                             nh3eTexFits,eTex_lim,region,plot_param,ax)
    plot_abundance_compare_panel_22(nh3ColFits,herschelNH2File,nh3TexFits,0,
                                    nh3eTexFits,eTex_lim,nh322Fits,nh322RmsFits,region,plot_param,ax)

plt.tight_layout()
fig.savefig('figures/DR1_XNH3_sn22_lim.pdf'.format(Tex_lim))
plt.close('all')
