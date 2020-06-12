# =============================================================================
# Imports
# =============================================================================
import ee

from satellite_image_operations import s2_band_dict_median, s2_band_dict

ee.Initialize()
from data import hcs_database as hcs_db
import satellite_image_operations as sat_ops
import pandas as pd
import requests
import zipfile
import timer

# =============================================================================
# Define paths
# =============================================================================
in_path = 'users/rheilmayr/indonesia/'
out_path = 'C:\\Users\\ME\\Dropbox\\HCSproject\\data\\PoC'

# =============================================================================
# Define date range
# =============================================================================
year='ALL'
date_start = ee.Date.fromYMD(2014, 1, 1)
date_end = ee.Date.fromYMD(2015, 12, 31)

# =============================================================================
# Load study data
# =============================================================================


# sites = ['app_jambi', 'app_oki', 'app_kaltim', 'app_kalbar',
#         'app_muba',
     #   'app_riau'
#         'crgl_stal', 'gar_pgm', 'nbpol_ob', 'wlmr_calaro']
sites = ['gar_pgm',
    'app_riau',
'app_kalbar',
        'app_kaltim',
     'app_jambi',
'app_oki',
       'crgl_stal'
    ]

feature_dict = {}
for site in sites:
    strata_img = ee.Image(hcs_db.rasters[site])
    geometry = strata_img.geometry()
    feature = ee.Feature(geometry)
    feature_dict[site] = feature
fc = ee.FeatureCollection(list(feature_dict.values()))
all_study_area = fc.geometry().bounds()
all_json_coords = all_study_area.getInfo()['coordinates']

# =============================================================================
# Get Sent2 Ndvi
# =============================================================================
def getYearlyNdvi_s2():
    yearly_ndvis = ee.Image()
    for year in range(2016, 2019):
        #year="ALL"
        name = "s2_ndvi_" + str(year)
        print("NAME:   ", name)
        date_start = ee.Date.fromYMD(year, 1, 1)
        date_end = ee.Date.fromYMD(year, 12, 31)
        sentinel2 = ee.ImageCollection('COPERNICUS/S2')
        sentinel2 = sentinel2.filterDate(date_start, date_end)

        # Pre-filter to get less cloudy granules.
        sentinel2 = sentinel2.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 35))
        sentinel2 = sentinel2.filterBounds(all_study_area)
        print('sentinel2 size:  ', sentinel2.size().getInfo())
        if (sentinel2.size().lt( ee.Number(1)).getInfo() ):
            continue
        sentinel2 = sentinel2.map(sat_ops.maskS2clouds)
        sentinel2 = sentinel2.map(sat_ops.addNDVI_s2)
        ndviMean = sentinel2.select('NDVI').max()
        ndviMean = ndviMean.rename(name+"_max")
        ndviVar = sentinel2.select('NDVI').reduce(ee.Reducer.stdDev())
        ndviVar = ndviVar.rename(name + "_var")
        yearly_ndvis = yearly_ndvis.addBands(ndviMean)#.addBands(ndviVar)
    return yearly_ndvis


# =============================================================================
# Get Landsat5 NDVI
# =============================================================================
def getYearlyNdvi_L5():
    yearly_ndvisL5 = ee.Image()
    for year in range(1999, 2012):
        #year="ALL"
        name = "ls5_ndvi_" + str(year)
        print("NAME:   ", name)
        date_start = ee.Date.fromYMD(year, 1, 1)
        date_end = ee.Date.fromYMD(year, 12, 31)
        l5 = ee.ImageCollection('LANDSAT/LT05/C01/T1_SR')
        l5 = l5.filterDate(date_start, date_end)

        # Pre-filter to get less cloudy granules.
        #sentinel2 = sentinel2.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 45))
        l5 = l5.filterBounds(all_study_area)
        print('L5 size:  ', l5.size().getInfo())
        print('L5 boolean eval: ', l5.size().lt( ee.Number(1)).getInfo() )
        if (l5.size().lt( ee.Number(1)).getInfo() ):
            continue
        #l5 = l5.map(sat_ops.maskCloudsL5)
        l5 = l5.map(sat_ops.addNDVI_l5)
        ndviMax = l5.select('NDVI').max()
        ndviMax = ndviMax.rename(name+"_max")
        ndviVar = l5.select('NDVI').reduce(ee.Reducer.stdDev())
        ndviVar = ndviVar.rename(name + "_var")
        yearly_ndvisL5 = yearly_ndvisL5.addBands(ndviMax)#.addBands(ndviVar)
    yearly_ndvisL5 = yearly_ndvisL5.set('SENSOR_ID', 'TM');
    return yearly_ndvisL5



# =============================================================================
# Get Landsat8 NDVI
# =============================================================================
def getYearlyNdvi_L8():
    yearly_ndvisL8 = ee.Image()
    for year in range(2011, 2019):
        #year="ALL"
        name = "ls8_ndvi_" + str(year)
        print("NAME:   ", name)
        date_start = ee.Date.fromYMD(year, 1, 1)
        date_end = ee.Date.fromYMD(year, 12, 31)
        l8 = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
        l8 = l8.filterDate(date_start, date_end)
        # Pre-filter to get less cloudy granules.
        #sentinel2 = sentinel2.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 45))
        l8 = l8.filterBounds(all_study_area)
        print('L8 size:  ',l8.size().getInfo() )
        if (l8.size().lt( ee.Number(1)).getInfo() ):
            continue
        l8 = l8.map(sat_ops.maskCloudsLandsat8)
        l8 = l8.map(sat_ops.addNDVI_l8)
        ndviMax = l8.select('NDVI').max()
        ndviMax = ndviMax.rename(name+"_max")
        ndviVar = l8.select('NDVI').reduce(ee.Reducer.stdDev())
        ndviVar = ndviVar.rename(name + "_var")
        yearly_ndvisL8 = yearly_ndvisL8.addBands(ndviMax)#.addBands(ndviVar)
    return yearly_ndvisL8

# =============================================================================
# SRTM
# =============================================================================
def getDEM(all_study_area):
    dem = ee.Image("USGS/SRTMGL1_003").clip(all_study_area)
    elevation = dem.select('elevation');
    slope = ee.Terrain.slope(elevation).rename('slope');
    elevation = elevation.unitScale(-30, 3300)
    slope = slope.unitScale(0, 80)
    dem = elevation.addBands(slope)

   # dem = ee.Image(dem.addBands(ee.Terrain.aspect(elevation).rename('aspect')));
  #  return dem
    return dem
# =============================================================================
# Soil Great Groups
# =============================================================================

def getSoil(all_study_area):
    soil = ee.Image("OpenLandMap/SOL/SOL_GRTGROUP_USDA-SOILTAX_C/v01").clip(all_study_area)
    return soil


# =============================================================================
# Prep landsat data
# =============================================================================

def assemble_l8(study_area, year):
    date_start = ee.Date.fromYMD(year-1, 1, 1) # need more than 1 year of landsat to make usable composite
    date_end = ee.Date.fromYMD(year, 12, 31)
    #ic = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
    ic = ee.ImageCollection("LANDSAT/LC08/C01/T1_TOA")
    ic = ic.filterDate(date_start, date_end)
    ic = ic.filterBounds(study_area)
    ic=ic.filter(ee.Filter.lt('CLOUD_COVER', 60))
    ic_masked = ic.map(sat_ops.prep_ls8)
    clean_l8_img = ee.Image(ic_masked.qualityMosaic('EVI'))
    return(clean_l8_img)

# =============================================================================
# Prep landsat data
# =============================================================================
def get_water_mask(study_area):
    water = ee.ImageCollection('COPERNICUS/S1_GRD')
    date_start = ee.Date.fromYMD(2013, 1, 1)
    date_end = ee.Date.fromYMD(2019, 12, 31)
    water = water.filterDate(date_start, date_end)
    water = water.filter(ee.Filter.eq('instrumentMode', 'IW'))
    water = water.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    water = water.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    water = water.filterBounds(study_area)
    water = water.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'));
    mask = ee.Image(sat_ops.prep_sar(water)).select(['VH'])
    return mask.gt(-19.0)

# =============================================================================
# Prep SAR data
# =============================================================================
# radarCollectionByYear = ee.ImageCollection(ee.List.sequence(2014,2018,1).map(prep_sar))
def assemble_radar_data(study_area, year):
    radar_composite = ee.Image()
    #for year in range(2015, 2016):
    sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD')
    date_start = ee.Date.fromYMD(year, 1, 1)
    date_end = ee.Date.fromYMD(year, 12, 31)
    print('RADAR YEAR:  ', year)
    sentinel1 = sentinel1.filterDate(date_start, date_end)
    sentinel1 = sentinel1.filter(ee.Filter.eq('instrumentMode', 'IW'))
    sentinel1 = sentinel1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    sentinel1 = sentinel1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    sentinel1 = sentinel1.filterBounds(study_area)
    sentinel1 = sentinel1.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'));
    myRadar = ee.Image(sat_ops.prep_sar(sentinel1)).select([ 'VV_0', 'VH_0'])

    return(myRadar)
#radar_composite = ee.Image(sat_ops.prep_sar(sentinel1))
# =============================================================================
# Prep Sentinel-2 data
# =============================================================================
def assemble_sentinel_data(study_area, year):
    sentinel2 = ee.ImageCollection('COPERNICUS/S2')
    date_start = ee.Date.fromYMD(year, 1, 1)
    date_end = ee.Date.fromYMD(year, 12, 31)
    sentinel2 = sentinel2.filterDate(date_start, date_end)
    # Pre-filter to get less cloudy granules.
    sentinel2 = sentinel2.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
    sentinel2 = sentinel2.filterBounds(study_area)
    #ndvis_s2 = getYearlyNdvi_s2()
    #ndvis_l8 = getYearlyNdvi_L8()
    #ndvis_l5 = getYearlyNdvi_L5()
    #elevation = getDEM().select('elevation');
    #slope = ee.Terrain.slope(elevation);
    #dem = ee.Image(elevation.addBands(slope));
    sentinel2_masked = sentinel2.map(sat_ops.prep_s2)
    #old_names = list(s2_band_dict_median.keys())
   # new_names = list(s2_band_dict_median.values())
    #clean_s2_img_med=ee.Image(sentinel2_masked.median()).select(old_names, new_names)
    old_names = list(s2_band_dict.keys())
    new_names = list(s2_band_dict.values())
    clean_s2_img_green = sentinel2_masked.qualityMosaic('EVI').select(old_names, new_names)
    return clean_s2_img_green


# =============================================================================
# Create site-level images for classification with reclassed strata and landsat data
# =============================================================================



def downloadLandcoverFiles(site):
    prefix = site + '_remap_3class'
    # fd.close()
    filename2 = out_path + '/' + site + '/' + prefix + '.zip'
    url = strata_remapped.clip(geometry).getDownloadURL({'name': prefix, 'crs': 'EPSG:4326', 'scale': 30})
    r = requests.get(url, stream=True)
    with open(filename2, 'wb') as fd:
         for chunk in r.iter_content(chunk_size=1024):
             fd.write(chunk)
    # # Extract the GeoTIFF for the zipped download
    # print(filename1)
    fd.close()
    z = zipfile.ZipFile(filename2)
    z.extractall(path=out_path + '/' + site)

    prefix = site + '_all_class'
    # fd.close()
    filename2 = out_path + '/' + site + '/' + prefix + '.zip'
    url = strata_img.clip(geometry).getDownloadURL({'name': prefix, 'crs': 'EPSG:4326', 'scale': 30})
    r = requests.get(url, stream=True)
    with open(filename2, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=1024):
            fd.write(chunk)
    # # Extract the GeoTIFF for the zipped download
    # print(filename1)
    fd.close()
    z = zipfile.ZipFile(filename2)
    z.extractall(path=out_path + '/' + site)


bands = list(sat_ops.l8_band_dict.values()) + list(['ndvi_l8'])
bands.extend(list(['remapped']))
bands.extend(list(sat_ops.s1_band_dict.values()))
#bands.extend(list(sat_ops.s2_band_dict.values()))

print(bands)
img_dict = dict.fromkeys(sites, 0)
if __name__ == "__main__":
    key_csv = '/Users/ME/Dropbox/HCSproject/data/strata_key.csv'
    key_df = pd.read_csv(key_csv)
    from_vals = list(key_df['project_code'].astype(float).values)
    to_vals = list(key_df['code_3class'].astype(float).values)
    for site in sites:
        strata_img = ee.Image(hcs_db.rasters[site])
        strata_remapped = strata_img.remap(from_vals, to_vals, 4)
        strata_img = strata_img.remap(from_vals, from_vals)
        geometry = strata_img.geometry()
        coords = geometry.coordinates()
        json_coords = coords.getInfo()
        strata_remapped = strata_remapped.int()
        strata_img = strata_img.int()
        images = {
          #  'landsat': clean_l8_img,
              #  '_max_s2': clean_s2_img_green,
        #    '_median_s2': clean_s2_img_med,
             #    '_S2_ndvi': ndvis,#S,
        #    '_L8_ndvi':ndvis_l8,
         #   '_l5_ndvi': ndvis_l5
        #       '_radar': radar_composite ,# 'class': strata_img
       #    '_dem': dem
            }
        #downloadLandcoverFiles(site)
        for key, value in images.items():
            prefix = site + key
            print('prefix:  ', prefix)
            url = value.clip(geometry).getDownloadURL({'name': prefix, 'crs': 'EPSG:4326', 'scale': 30})
            filename = out_path + '\\' + site + '\\in\\' + prefix + '.zip'
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
            z.extractall(path=out_path + '\\' + site + '\\in')

    # # print(clean_l8_img)
    # landsat_img = clean_l8_img.addBands(radar_composite) #.addBands(clean_s2_img)
    #     #.addBands(radar_composite)\
    #
    # # class_img = clean_s2_img
    # img_dict[site] = ee.Image(class_img).select(bands)
    # prefix = site + '_input'
    # url = class_img.clip(geometry).getDownloadURL({'name': prefix, 'crs': 'EPSG:4326', 'scale': 30})
    # # print(url)
    #
    # filename1 = out_path + '\\' + site + '\\in\\' + prefix + '.zip'
    # r = requests.get(url, stream=True)
    # with open(filename1, 'wb') as fd:
    #     for chunk in r.iter_content(chunk_size=1024):
    #         fd.write(chunk)
    #

    # # Extract the GeoTIFF for the zipped download
    # z = zipfile.ZipFile(filename2)
    # z.extractall(path=out_path + '/' + site)



