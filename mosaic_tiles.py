import rasterio
from rasterio.merge import merge
import satellite_image_operations as sat_ops
import glob
import os


#my_dict = sat_ops.s1_band_dict.copy()
my_dict = sat_ops.l8_band_dict.copy()
#my_dict = sat_ops.soil_band_dict.copy()
#my_dict = sat_ops.dem_band_dict.copy()
#my_dict=sat_ops.s2_band_dict.copy()
#my_dict.update(sat_ops.s2_band_dict)
#my_dict.update(sat_ops.dem_band_dict)
print('BANDS:  ',my_dict.values())
island='Kalimantan'
years=[2015,2019]#,2017]
for year in years:
    dirpath = r'/scratch/hcs_classifier/data/concession/' + island + '/in/' + str(year)
    out_fp = r'/scratch/hcs_classifier/data/concession/' + island + '/out/' + str(year) + '/'
    for band in my_dict.values():
        search_criteria='*'+band+'.tif'
        out_file = band+'.tif'
        print(search_criteria)
        q = os.path.join(dirpath, search_criteria)
        print(q)
        image_files = glob.glob(q)
        out = os.path.join(out_fp, out_file)
        #print(image_files)
        src_files_to_mosaic = []
        for f in image_files:
            src = rasterio.open(f)
            src_files_to_mosaic.append(src)

        mosaic, out_trans = merge(src_files_to_mosaic)
        out_meta = src.meta.copy()
        # Update the metadata
        out_meta.update({"driver": "GTiff",
             "height": mosaic.shape[1],
             "width": mosaic.shape[2],
             "transform": out_trans
                         }
        )
        with rasterio.open(out, "w", **out_meta) as dest:
             dest.write(mosaic)