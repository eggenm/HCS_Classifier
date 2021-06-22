# =============================================================================
# Imports
# =============================================================================
import ee
from satellite_image_operations import s2_band_dict_median, s2_band_dict
ee.Initialize()
import ingest_google_data as ingest
import dirfuncs
import os
import glob
import time

#================================================================================
# Parameters
#================================================================================
#A shape file - study area
lon_edge=2
# lon_end=119
# lat_start = -5
# lat_end = 5
#lat_edge = 2.5 #PAPUA
lat_edge = 2
#site = 'Kalimantan'
start = 1
end = 50
years= [#2017,2015,
     2019]
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
    #sentinel = ingest.assemble_sentinel_data(all_study_area, year)
    l8 = ingest.assemble_l8(all_study_area, year)
    #dem = ingest.getDEM(all_study_area)

    images = {
       #   '_greenest': sentinel,
        #  '_radar': radar,  # 'class': strata_img,
        '_median_l8': l8,
       # '_dem':dem

    }
    for key, value in images.items():

        print(value.bandNames().getInfo())
        for band in value.bandNames().getInfo():
                for geometry in polygons:
                    cell = int(geometry.get('label').getInfo())
                    if cell < start or cell > end:
                        continue
                    else:
                        prefix =   site + '/' + str(year) + '/' + site + key + '_'+str(i)+'_' + str(cell) + '_' + band
                        print('prefix:  ', prefix)
                        myimage = ee.ImageCollection(value).filterBounds(geometry.geometry()).first().select(band).clip(geometry)
                        filename = out_path + site + '/in/' + str(year) + '/' + prefix + '.zip'
                        failed = 0
                        while(failed<12):
                            try:
                                    task = ee.batch.Export.image.toCloudStorage(image=myimage, fileNamePrefix =prefix , bucket='hcsa_forest_mapping_training_bucket',  crs='EPSG:4326', scale=30 )
                                    task.start()
                                    state = task.status()['state']
                                    while state in ['READY', 'RUNNING']:
                                        print(state + '...')
                                        state = task.status()['state']
                                        time.sleep(8)
                                    print('Done.', task.status())
                                    if(state== 'FAILED'):
                                        failed += 1
                                        print('*****Error on download-extract from google. Times failed: ', failed)
                                        time.sleep(10)  # wait for 5 seconds if we are having trouble getting file from GEE
                                        if failed >= 5:
                                            raise TimeoutError
                                    failed = 99
                            except Exception as ex:
                                failed +=1
                                time.sleep(10)#wait for 5 seconds if we are having trouble getting file from GEE
                                if failed>=5:
                                    raise TimeoutError

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

    #Papua
     site = 'Papua'
     polygons = get_grid_polygons(129, 142, -10, 0)
     for year in years:
        download_data(polygons, 77, year)
        cleanup_files(year)


     ##KALIMANTAN
         # site = 'Kalimantan'
         # polygons = get_grid_polygons(107, 119, -5,7)
         # for year in years:
         #    download_data(polygons, 33, year)
         #    cleanup_files(year)

#
# ##SUMATRA
#          site = 'Sumatra'
#
#          polygons = get_grid_polygons(95, 109, -6,6)
#          for year in years:
#            download_data(polygons, 44, year)
         #polygons = get_grid_polygons(107, 109, -6, 6)
         #for year in years:
          # download_data(polygons, 55, year)


