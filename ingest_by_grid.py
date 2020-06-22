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
import os
import glob
import time
from data import hcs_database as hcs_db
import satellite_image_operations as sat_ops
import pandas as pd

#================================================================================
# Parameters
#================================================================================
#A shape file - study area
#Get a bounding box
# lon_start=107
lon_edge=.5
# lon_end=119
# lat_start = -5
# lat_end = 5
lat_edge = .5
#site = 'Kalimantan'
years = [2016, 2017, 2018, 2019]
start = 1
#years= [2017,2018,2019]
site = 'None'
out_path = dirfuncs.guess_data_dir()
#Take a set of years
#Take a set of bands
# take a step

# for each grid cell

    # for each band
def get_grid_polygons(lon_start, lon_end, lat_start,lat_end):
    print(lon_start, lon_end, lat_start,lat_end)
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
    return(polys)


def download_data(polys,i, year):
    fc = ee.FeatureCollection(polys)
    all_study_area = fc.geometry().bounds()
    #radar = ingest.assemble_radar_data(all_study_area, year)
    sentinel = ingest.assemble_sentinel_data(all_study_area, year)
    #l8 = ingest.assemble_l8(all_study_area, year)
    #dem = ingest.getDEM(all_study_area)
  #  soil = ingest.getSoil(all_study_area)
  #  water_mask = ingest.get_water_mask(all_study_area)

    images = {
          '_greenestwCDI': sentinel,
      #  '_radar': radar,  # 'class': strata_img,
      #  '_greenestw_mask2': l8,
      #  '_dem':dem
     #   '_soil': soil
       # '_watermask': water_mask

    }
    for key, value in images.items():
        for band in value.bandNames().getInfo():
            print(band)

            for geometry in polygons:
                cell = int(geometry.get('label').getInfo())
                if cell < start:
                    continue
                else:
                    prefix = site + key + '_'+str(i)+'_' + str(cell) + '_' + band
                    print('prefix:  ', prefix)
                    url = value.select(band).clip(geometry).getDownloadURL(
                        {'name': prefix, 'crs': 'EPSG:4326', 'scale': 30})
                    filename = out_path + site + '/in/' + str(year) + '/' + prefix + '.zip'
                    print(url)
                    failed = 0
                    while(failed<12):
                        try:
                            with timer.Timer() as t:
                                r = requests.get(url, stream=True)
                                with open(filename, 'wb') as fd:
                                    for chunk in r.iter_content(chunk_size=1024):
                                        fd.write(chunk)
                                fd.close()
                                z = zipfile.ZipFile(filename)
                                z.extractall(path=out_path + '/' + site + '/in/' + str(year))
                                failed = 99
                        except:
                            failed +=1
                            print('*****Error on download-extract from google. Times failed: ', failed)
                            time.sleep(10)#wait for 5 seconds if we are having trouble getting file from GEE
                            if failed==11:
                                raise TimeoutError
                        finally:
                            print('Request-Extract took %.03f sec.' % t.interval)


def cleanup_files(year):
    # Get a list of all the file paths that ends with .txt from in specified directory
    files = out_path + site + '/in/' + str(year) + '/*.zip'
    fileList = glob.glob(files)

    # Iterate over the list of filepaths & remove each file.
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)


if __name__ == "__main__":
     #KALIMANTAN
         site = 'Kalimantan'
         polygons = get_grid_polygons(107, 119, -5,5)
         for year in years:
            download_data(polygons, 33, year)
            cleanup_files(year)
#
#
# ##SUMATRA
         site = 'Sumatra'
         polygons = get_grid_polygons(95, 107, -6,6)
         download_data(polygons, 44)
         cleanup_files()
       #polygons = get_grid_polygons(107, 110, -6, 6)
       #for year in years:
           #download_data(polygons, 55)
           #cleanup_files()
#something
#      polygons = get_grid_polygons(98, 102, -6, 4)
       #for year in years:

           # download_data(polygons, 22 )
           # cleanup_files()
        # polygons = get_grid_polygons(98, 99, 4, 5)
        #for year in years:

           # download_data(polygons, 12)
           # cleanup_files()
       # polygons = get_grid_polygons(106, 109, -4, 1)
       # for year in years:
           # download_data(polygons, 13)
           # cleanup_files()

