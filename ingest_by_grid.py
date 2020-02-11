# =============================================================================
# Imports
# =============================================================================
import ee
from satellite_image_operations import s2_band_dict_median, s2_band_dict
ee.Initialize()
import ingest_google_data as ingest
import zipfile
import timer
import requests
import dirfuncs
from data import hcs_database as hcs_db
import satellite_image_operations as sat_ops
import pandas as pd

#================================================================================
# Parameters
#================================================================================
#A shape file - study area
#Get a bounding box
# lon_start=110#108
# # lon_edge=2
# # lon_end=116#120
# # lat_start = -2
# # lat_end = 2
# # lat_edge = 2
# # site = 'Kalimantan'
year = 2015
lon_start=100
lon_edge=2
lon_end=106
lat_start = -6
lat_end = 1
lat_edge = 1
site = 'Sumatra'
out_path = dirfuncs.guess_data_dir()
#Take a set of years
#Take a set of bands
# take a step

# for each grid cell

    # for each band
polys = [];
lon = lon_start
cell_id=0
while lon < lon_end :
  x1 = lon;
  x2 = lon + lon_edge;
  lon += lon_edge
  lat = lat_start
  while lat < lat_end :
    y1 = lat;
    y2 = lat + lat_edge;
    #cell_id = str(x1) + '_' + str(y1)
    cell_id = cell_id + 1
    lat += lat_edge
    print('x1: ' , x1, '  y1: ', y1, 'x2: ' , x2, '  y2: ', y2)
    polys.append(ee.Feature(ee.Geometry.Rectangle(x1, y1, x2, y2) , {'label': str(cell_id)}));

fc = ee.FeatureCollection(polys)
all_study_area = fc.geometry().bounds()
radar  = ingest.assemble_radar_data(all_study_area, year)
sentinel = ingest.assemble_sentinel_data(all_study_area, year)
l8 = ingest.assemble_l8(all_study_area, year)


images = {
              #  '_greenest': sentinel,
               '_radar': radar ,# 'class': strata_img,
                '_greenest':l8
            }
for key, value in images.items():
    for band in value.bandNames().getInfo():
        print(band)
        for geometry in polys:

            prefix = site + key +'_'+ geometry.get('label').getInfo() + '_' + band
            print('prefix:  ', prefix)
            url = value.select(band).clip(geometry).getDownloadURL({'name': prefix, 'crs': 'EPSG:4326', 'scale': 30})
            filename = out_path  + site + '/in/'+ str(year) + '/' + prefix + '.zip'
            print(url)
            try:
                with timer.Timer() as t:
                    r = requests.get(url, stream=True)
                    with open(filename, 'wb') as fd:
                        for chunk in r.iter_content(chunk_size=1024):
                            fd.write(chunk)
                    fd.close()
            finally:
                print('Request took %.03f sec.' % t.interval)
            z = zipfile.ZipFile(filename)
            z.extractall(path=out_path + '/' + site + '/in/'+ str(year) )