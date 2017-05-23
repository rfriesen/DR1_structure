'''
Need to mask Orion Nebula where H2 fits are not reliable
'''
from astropy.io import fits
import astropy.units as u
import astropy.constants as c
from astropy.coordinates import Angle, SkyCoord
from astropy.coordinates import ICRS
from regions import EllipseSkyRegion
from astropy.wcs import WCS
import numpy as np

# Use regions to get ellipse covering ONC
onc_coords = SkyCoord('5h35m13.9s -5d22m51s',ICRS)
a_maj = 310.*u.arcsec
a_min = 210.*u.arcsec
a_pa  = 105.*u.deg
ellipse_sky = EllipseSkyRegion(onc_coords,a_maj, a_min, angle=a_pa)
# Use wcs from Orion A map to convert to pixels, and mask:
OrionA_image = 'nh2_regridded/OrionA_NH2_regrid.fits'
hdulist = fits.open(OrionA_image)
wcs = WCS(hdulist[0].header)
ellipse_pix = ellipse_sky.to_pixel(wcs)
# First plot on Orion data to see if have the mask right. Looks good. 
fig, ax = plt.subplots(1,1)
ax.imshow(np.log10(hdulist[0].data),origin='lower',cmap='Greys_r')
patch = ellipse_pix.as_patch(facecolor='none',edgecolor='red',lw=2)
ax.add_patch(patch)
fig.savefig('figures/OrionA_mask.pdf')
plt.close('all')

# Then make mask.
mask = ellipse_pix.to_mask()
full_mask = mask.to_image(hdulist[0].data.shape)
# Invert
full_mask = 1 - full_mask
# Multiply with N(H2) data
masked_data = hdulist[0].data * full_mask
masked_hdulist = hdulist
masked_hdulist[0].data = masked_data
masked_hdulist.writeto('nh2_regridded/OrionA_NH2_regrid_masked.fits',
                       overwrite=True)
# Multiply with Td data
hdulist = fits.open('Td_regridded/OrionA_Td_regrid.fits')
masked_data = hdulist[0].data * full_mask
masked_hdulist = hdulist
masked_hdulist[0].data = masked_data
masked_hdulist.writeto('Td_regridded/OrionA_Td_regrid_masked.fits',
                       overwrite=True)
