# =============================================================================
# Imports
# =============================================================================
import ee
ee.Initialize()
import satellite_image_operations as sat_ops


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
        date_end = ee.Date.fromYMD(year, 6, 30) #as this due to fires?
        l8 = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
        l8 = l8.filterDate(date_start, date_end)
        # Pre-filter to get less cloudy granules.
        #sentinel2 = sentinel2.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 45))
        l8 = l8.filterBounds(all_study_area)
        print('L8 size:  ',l8.size().getInfo() )
        if (l8.size().lt( ee.Number(1)).getInfo() ):
            continue
        l8 = l8.map(sat_ops.maskCloudsLandsat8)
        l8 = l8.map(sat_ops.add_EVI2_l8)
        ndviMax = l8.select('EVI').max()
        ndviMax = ndviMax.rename(name+"_max")
       # ndviVar = l8.select('NDVI').reduce(ee.Reducer.stdDev())
       # ndviVar = ndviVar.rename(name + "_var")
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
    return dem


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
    ic=ic.filter(ee.Filter.lt('CLOUD_COVER', 75))

    #composite = ee.Algorithms.Landsat.simpleComposite(ic)
    ic = ic.map(sat_ops.add_cloud_score_mask)
    ic = ic.map(sat_ops.add_EVI2_l8)
    ic = ic.map(sat_ops.add_tassle_cap_l8)
    #ic_masked = ic.map(sat_ops.prep_ls8)
    clean_l8_img = ee.Image(ic.median())
    old_names = list(sat_ops.l8_band_dict.keys())
    new_names = list(sat_ops.l8_band_dict.values())
    clean_l8_img = clean_l8_img.select(old_names, new_names)
    return(clean_l8_img)



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
    myRadar = ee.Image(sat_ops.prep_sar(sentinel1)).select([
        'VV_0',
       'VH_0',
    'VV',
     'VH',
        'VV_2',
         'VH_2'
    ])

    return(myRadar)






