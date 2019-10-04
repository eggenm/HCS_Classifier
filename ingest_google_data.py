# =============================================================================
# Imports
# =============================================================================
import ee
ee.Initialize()
import numpy as np
import hcs_database as hcs_db
import satellite_image_operations as sat_ops
import pandas as pd
import requests
import zipfile

# =============================================================================
# Define paths
# =============================================================================
in_path = 'users/rheilmayr/indonesia/'
out_path = '/Users/ME/Dropbox/HCSproject/data/PoC'

# =============================================================================
# Define date range
# =============================================================================
year='ALL'
date_start = ee.Date.fromYMD(2014, 1, 1)
date_end = ee.Date.fromYMD(2018, 12, 31)

# =============================================================================
# Load study data
# =============================================================================
key_csv = '/Users/ME/Dropbox/HCSproject/data/strata_key.csv'
key_df = pd.read_csv(key_csv)
from_vals = list(key_df['project_code'].astype(float).values)
to_vals = list(key_df['code_simpl'].astype(float).values)

# sites = ['app_jambi', 'app_oki', 'app_kaltim', 'app_kalbar',
#         'app_muba',
#         'app_riau',
#         'crgl_stal', 'gar_pgm', 'nbpol_ob', 'wlmr_calaro']
sites = ['app_kaltim', 'gar_pgm']

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
# Prep landsat data
# =============================================================================
ic = ee.ImageCollection('LANDSAT/LC08/C01/T2_SR')
ic = ic.filterDate(date_start, date_end)
ic = ic.filterMetadata(name='WRS_ROW', operator='less_than', value=120)
ic = ic.filterBounds(all_study_area)
ic_masked = ic.map(sat_ops.prep_ls8)
clean_l8_img = ee.Image(ic_masked.qualityMosaic('ndvi_l8'))
print(clean_l8_img.bandNames().getInfo())

# =============================================================================
# Prep SAR data
# =============================================================================
# radarCollectionByYear = ee.ImageCollection(ee.List.sequence(2014,2018,1).map(prep_sar))
sentinel1 = ee.ImageCollection('COPERNICUS/S1_GRD')
sentinel1 = sentinel1.filterDate(date_start, date_end)
sentinel1 = sentinel1.filter(ee.Filter.eq('instrumentMode', 'IW'))
sentinel1 = sentinel1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
sentinel1 = sentinel1.filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
sentinel1 = sentinel1.filterBounds(all_study_area)
sentinel1 = sentinel1.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'));

radar_composite = ee.Image(sat_ops.prep_sar(sentinel1))
# =============================================================================
# Prep Sentinel-2 data
# =============================================================================
sentinel2 = ee.ImageCollection('COPERNICUS/S2')
sentinel2 = sentinel2.filterDate(date_start, date_end)
# Pre-filter to get less cloudy granules.
sentinel2 = sentinel2.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 35))
sentinel2 = sentinel2.filterBounds(all_study_area)
sentinel2_masked = sentinel2.map(sat_ops.prep_s2)
# clean_s2_img=ee.Image(sentinel2_masked.median())
clean_s2_img = sentinel2_masked.qualityMosaic('ndvi_s2')

# =============================================================================
# Create site-level images for classification with reclassed strata and landsat data
# =============================================================================
bands = list(sat_ops.l8_band_dict.values()) + list(['ndvi_l8'])
bands.extend(list(['remapped']))
bands.extend(list(sat_ops.s1_band_dict.values()))
bands.extend(list(sat_ops.s2_band_dict.values()))

print(bands)
img_dict = dict.fromkeys(sites, 0)
for site in sites:
    strata_img = ee.Image(hcs_db.rasters[site])
    strata_img = strata_img.remap(from_vals, to_vals, 4)
    geometry = strata_img.geometry()
    coords = geometry.coordinates()
    json_coords = coords.getInfo()
    strata_img = strata_img.int()

    # print(clean_l8_img)
    class_img = clean_l8_img.addBands(radar_composite).addBands(clean_s2_img)

    # class_img = clean_s2_img
    img_dict[site] = ee.Image(class_img).select(bands)
    prefix = site + '_input'
    url = class_img.clip(geometry).getDownloadURL({'name': prefix, 'crs': 'EPSG:4326', 'scale': 30})
    # print(url)

    filename1 = out_path + '/' + site + '/' + prefix + '.zip'
    r = requests.get(url, stream=True)
    with open(filename1, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=1024):
            fd.write(chunk)

    prefix = site + '_classRemap'
    filename2 = out_path + '/' + site + '/' + prefix + '.zip'
    url = strata_img.clip(geometry).getDownloadURL({'name': prefix, 'crs': 'EPSG:4326', 'scale': 30})
    r = requests.get(url, stream=True)
    with open(filename2, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=1024):
            fd.write(chunk)
    # Extract the GeoTIFF for the zipped download
    z = zipfile.ZipFile(filename1)
    z.extractall(path=out_path + '/' + site)
    # Extract the GeoTIFF for the zipped download
    z = zipfile.ZipFile(filename2)
    z.extractall(path=out_path + '/' + site)
