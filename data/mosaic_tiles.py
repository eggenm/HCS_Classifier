import rasterio
from rasterio.merge import merge
from rasterio.plot import show
import glob
import os

dirpath = r'C:\\Users\\ME\\Dropbox\\HCSproject\\data\\PoC\\Sumatra\\in\\2015'
out_fp = r'C:\\Users\\ME\\Dropbox\\HCSproject\\data\\PoC\\Sumatra\\out\\SumatraMosaicBlue.tif'
search_criteria = "*blue_max*.tif"
q = os.path.join(dirpath, search_criteria)
image_files = glob.glob(q)

src_files_to_mosaic = []
for f in image_files:
    src = rasterio.open(f)
    src_files_to_mosaic.append(src)

print(src_files_to_mosaic)
mosaic, out_trans = merge(src_files_to_mosaic)
out_meta = src.meta.copy()
# Update the metadata
out_meta.update({"driver": "GTiff",
     "height": mosaic.shape[1],
     "width": mosaic.shape[2],
     "transform": out_trans,
   #  "crs": "+proj=utm +zone=35 +ellps=GRS80 +units=m +no_defs "
                 }
)
with rasterio.open(out_fp, "w", **out_meta) as dest:
     dest.write(mosaic)