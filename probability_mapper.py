import dirfuncs
import glob
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
from osgeo import gdal,gdalconst
import rasterio as rio
import rasterio.warp

base_dir = dirfuncs.guess_data_dir()
concession = 'app_oki'
prob_file = 'prob_fileRF_x50_at_5__plusYearlyRadar.tif'

outtif = base_dir + concession + '/sklearn_test/' + prob_file

with rio.open(outtif) as img_src:
    img = img_src.read(1) # read the first band
    shape=img.shape
class_bins = [0, 0.6] #the first bin is forest and second non-forest
reclass1 = np.digitize(img,class_bins)
clas_img = (-1*(reclass1-2) * 255).astype('uint8') # 0 is non forest 255 is  forest
clas_img = Image.fromarray(clas_img)
#clas_img.show()
print(img.max())
result=img.copy()
result[result>-99999]=0

print(img.max())
with rio.open(outtif) as img_src:
    img2 = img_src.read(2) # read the first band

class_bins = [0, 0.35]
reclass2 = np.digitize(img2,class_bins)
clas_img = ((reclass2-1)* 255).astype('uint8') # 0 is non forest 255 is  forest
clas_img = Image.fromarray(clas_img)
#clas_img.show()
print(img.max())
with rio.open(outtif) as img_src:
    img3 = img_src.read(3) # read the first band
class_bins = [0, 0.35]
reclass3 = np.digitize(img3,class_bins)
clas_img = ((reclass3-1)* 255).astype('uint8') # 0 is non forest 255 is  forest
clas_img = Image.fromarray(clas_img)
#clas_img.show()
print(img.max())
#reclass4 = 1*(reclass1 == 0 * reclass2 == 0 * reclass3 == 0)


import matplotlib.pyplot as plt




final_image = -1*(reclass1-2) + ((reclass2-1) | (reclass3-1))
bins=[-999,0.1,1.1]
print(final_image.shape)
clas_img = ((final_image * 255)/2).astype('uint8')
clas_img = Image.fromarray(clas_img)
#clas_img.show()

print(img.max())
result[img2>=img3]=1
result[img3>img2]=2
result[img>0.55]=0
print(img.max())
referencefile = base_dir + concession + '/' + concession + '*remap*.tif'
out_clas_file = base_dir + concession + '/sklearn_test/prob_file_remap4.tif'
final_image=final_image[np.newaxis, :, :].astype(rio.int16)
result=result[np.newaxis, :, :].astype(rio.int16)
file_list = sorted(glob.glob(referencefile))
with rio.open(file_list[0]) as src:
    height = src.height
    width = src.width
    crs = src.crs
    transform = src.transform
    dtype = rio.int16
    count = 1
    with rio.open(out_clas_file, 'w', driver = 'GTiff',
                  height = height, width = width,
                  crs = crs, dtype = dtype,
                  count = count, transform = transform) as clas_dst:
        clas_dst.write(final_image)


clas_dst.close()