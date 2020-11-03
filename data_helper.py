# =============================================================================
# Imports
# =============================================================================
import pyproj
import dirfuncs
import os
import re
import glob
import pandas as pd
import numpy as np
import rasterio as rio
import rasterio.crs
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sklearn.metrics
from sklearn.metrics import f1_score
import ShapefileHelper as shapefilehelp
import data.hcs_database  as db
import itertools
from rasterio.mask import mask
from rasterio import Affine as A
from rasterio.warp import reproject, Resampling
import rioxarray as rx
import timer
import imagery_data
import itertools as it
from rioxarray import merge as rxmerge
import matplotlib.pyplot as plt
import satellite_image_operations as sat_ops
import sys

# =============================================================================
# Identify files
# =============================================================================
base_dir = dirfuncs.guess_data_dir()
pixel_window_size = 1
stackData = True
write_input_data = True
image_cache = imagery_data.Imagery_Cache.getInstance()


#classes = {1: "HCSA",
     #      0: "NA"}
sites = ['gar_pgm',
    'app_riau',
  'app_kalbar',
         'app_kaltim',
      'app_jambi',
 'app_oki',
        'crgl_stal'
    ]

bands_base=['blue_max', 'green_max', 'red_max', 'nir_max', 'swir1_max', 'swir2_max']

bands_historical=['ndvi_2013', 'ndvi_2014', 'ndvi_2015', 'ndvi_2011', 'ndvi_2010', 'ndvi_2009', 'ndvi_2008', 'ndvi_2007', 'ndvi_2006', 'ndvi_2005', 'ndvi_2004', 'ndvi_2003']

bands_median=['S2_blue_median', 'S2_green_median', 'S2_red_median', 'S2_nir_median', 'S2_nir2_median', 'S2_swir1_median', 'S2_swir2_median', 'S2_swir3_median', 'S2_vape_median']

bands_radar=['VH', 'VV']

bands_dem=['elevation', 'slope']

bands_extended=['rededge1', 'rededge2', 'rededge3']

bands_evi2_separate=['S2_red_max', 'S2_nir_max']

band_evi2 = ['EVI']

bands_evi2 = ['S2_red_max', 'S2_nir_max', 'EVI2_s2_max']

key_csv = base_dir + 'strata_key.csv'
key_df = pd.read_csv(key_csv)
from_vals = list(key_df['project_code'].astype(float).values)
to_vals = list(key_df['code_3class'].astype(float).values)
to_2class = list(key_df['code_simpl'].astype(float).values)
three_class_landcoverClassMap = dict( zip(from_vals,to_vals ))
two_class_landcoverClassMap = dict( zip(from_vals,to_2class ))
three_to_two_class_landcoverClassMap = dict( zip(to_vals,to_2class ))


classes = {
1:	'HDF',
2:	'MDF',
3:	'LDF',
4:	'YRF',
5:	'YS',
6:	'OL',
7:	'F',
8:	'E',
9:	'G',
10:	'NP',
11:	'OP',
12:	'DF',
13:	'C',
14:	'R',
15:	'RT',
16:	'W',
17:	'P',
18:	'SH',
19:	'AQ',
20:	'AG',
21:	'TP'
}


# =============================================================================
# FUNCTIONS:  Read and prep raster data
# =============================================================================
def return_window(img, i, j, n):
    """
    Parameters
    ----------
    array: np array
        Array of image to pull from

    i: int
        row location of center

    j: int
        column location of center

    n: int
        width of moving window

    Returns
    -------
    window: np array
        nxn array of values centered around pixel i,j
    """
    shift = (n - 1) / 2
    window = img[:, int(i - shift):int(i + shift + 1), int(j - shift):int(j + shift + 1)]
    return window


def gen_windows(array, n):
    """
    Parameters
    ----------
    array: np array
        Image from which to draw windows

    n: int
        width of moving window

    Returns
    -------
    windows: pandas dataframe
        df with ixj rows, with one column for every pixel values in nxn window
        of pixel i,j
    """
    try:
        with timer.Timer() as t:
            shape = array.shape
            print('SHAPE:  ',shape)
            start = int((n - 1) / 2)
            end_i = shape[1] - start
            end_j = shape[2] - start
            win_dict = {}
            for i in range(start, end_i):
                for j in range(start, end_j):
                    win_dict[(i, j)] = return_window(array, i, j, n)
            windows = pd.Series(win_dict)
            windows.index.names = ['i', 'j']
            index = windows.index
            windows = pd.DataFrame(windows.apply(lambda x: x.flatten()).values.tolist(), index=index)
    finally:
        print('gen_windows original request took %.03f sec.' % t.interval)
    return windows


def gen_windows2(array):
    try:
        with timer.Timer() as t:
            x=range(0, array.shape[1])
            y=range(0, array.shape[2])

            myfunc = lambda a: a.flatten()
            aList = [myfunc(array[i, :, :]) for i in range(0, array.shape[0])]
            full_index = pd.MultiIndex.from_product([x, y], names=['i', 'j'])
            i = 0
            x=False
            y=False
            windows = pd.DataFrame({i: aList[0]}, index=full_index)
            for i in range(1, len(aList)):
                print( 'gen_windows:  ', i, '  of  ', len(aList))
                temp = pd.DataFrame({i: aList[i]}, index=full_index)
                windows = windows.merge(temp, left_index=True, right_index=True, how='left')
                temp=False
            windows.index.names = ['i', 'j']
            aList=False
            array=False
    finally:
        print('gen_windows2 request took %.03f sec.' % t.interval)
    return windows



def get_classes(classImage, name):
    clas_dict = {}
    shape = classImage.shape
    try:
        with timer.Timer() as t:
            for i in range(classImage.shape[1]):
                for j in range(classImage.shape[2]):
                    clas_dict[(i, j)] = classImage[0, i, j]
            full_index = pd.MultiIndex.from_product([range(shape[1]), range(shape[2])], names=['i', 'j'])
            classes = pd.DataFrame({name: pd.Series(clas_dict)}, index=full_index)
    finally:
        print('get_classes Request took %.03f sec.' % t.interval)
    return classes


def combine_input_landcover(input, landcover_all):
    try:
        with timer.Timer() as t:
            data_df = landcover_all.merge(input, left_index=True, right_index=True, how='left')
            #data_df = landcover2.merge(data_df, left_index=True, right_index=True, how='left')
            data_df[data_df <= -999] = np.nan  #MEE 6-1-2020:  This used to be conditional on whether I was using the data in a training set or not. If I was doing it to make a class map then I did not do this
            #data_df = data_df.dropna()
            #print('*****data_df shape:  ', data_df.shape)
            return data_df
    finally:
        print('combine_input_landcover Request took %.03f sec.' % t.interval)


def scale_data(x):
    try:
        with timer.Timer() as t:
            print('DEPRECATED - SCALING DONE ON INGEST')
            #scaler = StandardScaler()
            #x_scaled = scaler.fit_transform(x.astype(np.float64))
            #return x_scaled
    finally:
        print('ScaleData Request took %.03f sec.' % t.interval)


def get_input_band(band, name, year):
    try:
        with timer.Timer() as t:
            print(band)
            image_cache = imagery_data.Imagery_Cache.getInstance()
            print('IMAGE CACHE SINGLETON: ',image_cache)
            if(db.data_context_dict!='supplementary_class'):
                image = image_cache.get_band_by_context_year(band, name, year)
            #else
               # image = image_cache.get_fixed_band
    finally:
        print('Get' , band , ' Request took %.03f sec.' % t.interval)
    return image


def trim_input_band_by_shape(input_raster, boundary):
    out_img, out_transform = mask(input_raster, shapes=boundary, crop=True)
    return out_img, out_transform


def reproject_match_input_band(band, island, year, bounding_raster):
    image2 = bounding_raster # rx.open_rasterio(bounding_raster)
    print('image2.shape:  ', image2.shape)
    # plt.figure()
    # image2.plot()
    # plt.show()
    image3 = get_input_band(band, island, year)
    print('image3.shape:  ', image3.shape)
    if(image3.dtype=='float64'):
        image3.data  = np.float32(image3)
    # plt.figure()
    # image3.plot()
    # plt.show()
    image3 = image3.rio.reproject_match(image2)
    print('image2.shape:  ', image2.shape)
    print('image3.shape:  ', image3.shape)
   # plt.figure()
   # destination.plot(robust=True)
  #  plt.show()

    image2=False
    #image3=False
    return image3

def write_concession_band(data_src, bounding_raster,  outtif):

    with bounding_raster as image:
        height = image.rio.height
        width = image.rio.width
        shape = image.rio.shape
        crs = image.rio.crs
        trans = image.transform
        with rasterio.open(outtif, 'w', driver = 'GTiff',
                      height = height, width = width,
                      crs = crs, dtype = data_src[0].dtype,
                      count = 1, transform = trans) as dst:
                   # dst.write_band(1, data_src[0])
            for ji, window in dst.block_windows(1):  # or re
               #  print('window.shape:  ', window.shape)
                 block = data_src[window.col_off:window.col_off+window.width,window.row_off:window.row_off+window.height ]#.read(window=window)
            #     if sum(sum(sum(~np.isnan(block)))) > 0:
                 dst.write(block , window=window)
        dst.close()




def get_feature_inputs(band_groups, bounding_box,  year, concession=None, filename=None):
    tif=''
    try:
        with timer.Timer() as t:
            print('Band_Groups:  ',band_groups)
            x = range(0, bounding_box.shape[1])
            y = range(0, bounding_box.shape[2])
            #array = [0 for x in range(len(band_groups))]
            #print('len(array):  ', len(array))
            image_cache = imagery_data.Imagery_Cache.getInstance()
            myfunc = lambda a: a.flatten()
            full_index = pd.MultiIndex.from_product([x, y], names=['i', 'j'])
            x = False
            y = False
            windows = pd.DataFrame()
            for i, band in enumerate(band_groups):
                context = db.data_context_dict[concession]
                try:
                    if(context=='supplementary_class'):
                        tif = image_cache.get_fixed_input_image_path(filename,  band)
                        file = glob.glob(tif)
                        out_img = rx.open_rasterio(file[0])

                    else:
                        out_img = image_cache.get_band_by_name_year(band, concession, year, context)

                except:
                        tif = image_cache.get_input_image_path(concession, year, context, band)
                        print('except: ', band , concession, context)
                        out_img = reproject_match_input_band(band, context, year, bounding_box)
                        if (write_input_data):
                            write_concession_band(out_img, bounding_box,  tif)
                #out_img = myfunc(out_img[0, :, :])
                out_img = np.asarray(out_img[0]).flatten()
                print('I:  ',i)
                #array[i] = np.asarray(out_img[0])  #WHY AM I DOING THIS, CANTA I JUST MAKE MY ARRAY HERE INSTAED OF 2 lines down??
                if windows.empty :
                    windows = pd.DataFrame({i: out_img}, index=full_index)
                else:
                    windows = windows.merge(pd.DataFrame({i: out_img}, index=full_index), left_index=True, right_index=True, how='left')

                out_img = False
            windows.index.names = ['i', 'j']
            full_index = False
            return windows
           # return np.asarray(array)
    finally:
        print('get_feature_inputs took %.03f sec.' % t.interval)



def get_concession_bands(bands, year, bounding_box, concession=None, filename = None):
    try:
        x=False
        with timer.Timer() as t:
            return(get_feature_inputs(bands, bounding_box,  year, concession, filename))
            #img = get_feature_inputs(bands, bounding_box,  year, concession)
            #x = gen_windows2(img)
    finally:
        #img=False
        print('get_concession_bands Request took %.03f sec.' % t.interval)
    #return x



def get_fixed_bands(bands, name, year, context):
    print('GETTING fixed class: ', name)
    try:
        x=pd.DataFrame()
        y = pd.DataFrame()
        with timer.Timer() as t:
            for file in image_cache.get_fixed_class_paths(name, context):
                class_image = rx.open_rasterio(file)
                band_data = get_concession_bands(bands, year, class_image, name, file)
                x = pd.concat([x,band_data], ignore_index=True)
                y = pd.concat([y,get_classes(class_image.data, 'clas')], ignore_index=True) #flatten?
            data = combine_input_landcover(x, y)

            #TODO, maybe for these each band could be cached as a csv?, y values are fixed , just make sure nodata is handled (discarded)
    finally:
        x=False
        y=False
        band_data=False
        class_image=False
        print('get_fixed_bands Request took %.03f sec.' % t.interval)
    return data




def get_input_data(bands, year, sites, get_predictor_data_only=False):
    data_by_site = dict()

    try:
        with timer.Timer() as t:
            for site in sites:
                print(site)
                type = db.data_context_dict[site]
                data = pd.DataFrame()
                if (type == 'supplementary_class'):
                    print('Getting fixed class data')
                    data = get_fixed_bands(bands, site, year, type)
                    # TODO get_fixed_bands
                    # get fixed bands will query a folder for all shape or rasters
                    # look for an input_sitename based on the classfilename
                    # make a y vector of class_id and nodatas
                    #
                else:
                    island = db.data_context_dict[site]
                    all_class = image_cache.get_class_by_concession_name(site)
                    #box = shapefilehelp.get_bounding_box_polygon(db.shapefiles[site])
                    x = get_concession_bands(bands, year, all_class, site)
                    if not get_predictor_data_only:

                        y = get_classes(all_class.data, 'clas')
                        data = combine_input_landcover(x, y)
                    elif get_predictor_data_only:
                        data = x
                data_by_site[site] = data
            all_class=False
            x=False
            y=False
    finally:
        print('get_input_data Request took %.03f sec.' % t.interval)
    return data_by_site

def get_large_area_input_data(study_area_base_raster, bands, island, year, name=None):
        try:
            with timer.Timer() as t:
                x = get_concession_bands(bands, year, study_area_base_raster, name)
                x = drop_no_data(x)
                #X_scaled_class = scale_data(x)
                return x
               # print('X_scaled_class.shape:  ', X_scaled_class.shape)
        finally:
            x = False
            print('Get Input Data Request took %.03f sec.' % t.interval)


def get_reference_raster_from_shape(shapefile, island, year):
    bounding = shapefilehelp.get_bounding_box_polygon(db.shapefiles[shapefile])
    outtif = get_input_band('nir_max', island, year)
    #out_img = reproject_match_input_band(outtif)
    out_img =outtif.rio.clip(bounding, outtif.rio.crs)
    outtif = False
    return out_img



def remove_low_occurance_classes( X, class_data):
    df= pd.DataFrame(data=[X, class_data])
    threshold = 10  # Anything that occurs less than this will be removed.
    df = df.groupby('clas').filter(lambda x: len(x) > threshold)

def map_to_3class(X):
    print(max(X))
    print(min(X))
    print(three_class_landcoverClassMap)
    return pd.Series(X).map(three_class_landcoverClassMap)

def map_to_2class(X):
    return pd.Series(X).map(two_class_landcoverClassMap)

def map_3_to_2class(X):
    return pd.Series(X).map(three_to_two_class_landcoverClassMap)

def trim_data(input):
    return input.groupby('clas').filter(lambda x: len(x) > 10000)

def trim_data2(input):
    #return input[input.clas.isin([21.0, 18.0, 7.0, 6.0, 4.0, 5.0, 20.0])]
    return input[np.logical_not(input.clas.isin([8.0]))]

def log_result():
    print('')

def score_model(y_test, yhat):
    show_results(y_test, yhat)
    f1 = f1_score(y_test, yhat, average='macro')
    f1_weighted = f1_score(y_test, yhat, average='weighted')
    return f1,f1_weighted

def show_results(y_test, y_hat):
    report = sklearn.metrics.classification_report(y_test, y_hat, output_dict=True)
    print(report)
    # export_report = pd.DataFrame(report).to_csv(r'C:\Users\ME\Desktop\export_report_riau.csv', index=None,
    #                           header=True)
    confMatrix = sklearn.metrics.confusion_matrix(y_test, y_hat)
    print(confMatrix)

def drop_no_data(data):
    try:
        with timer.Timer() as t:
            data[data <= -999] = np.nan
            data[data == 255] = np.nan
            data[data >= 9999] = np.nan
            return data.dropna()
    finally:
        print('Drop NoData Request took %.03f sec.' % t.interval)

#print(landcoverClassMap)
if __name__ == "__main__":
    concessions_csv = base_dir + 'concession_inventory.csv'
    con_df = pd.read_csv(concessions_csv)
    my_dict = sat_ops.s2_band_dict.copy()
    my_dict.update(sat_ops.s1_band_dict)
    my_dict.update(sat_ops.dem_band_dict)
    bands = my_dict.values()
    # for index, row in con_df.iterrows():
    #     print(row['app_key'], bool(row['ingest']), row['assessment_year'])
    #     if(bool(row['ingest'])):
    #         x = get_input_data(bands, str(int(row['assessment_year'])), [row['app_key']], False)


    print(con_df)
    #write_input_data=True
    x = get_input_data(bands, str(2019), ['Kalimantan'], True)
   # x = get_input_data(bands, str(2018), ['Kalimantan'], True)
  #  x = get_input_data([ 'elevation'],  str(2015), ['gar_pgm', 'Bumitama_PTGemilangMakmurSubur','PTAgroAndalan','PTMitraNusaSarana', 'Bumitama_PTDamaiAgroSejahtera']
   #                    , False )#,
   # x = get_input_data(['nir_max'], str(2019), ['impervious'], False)
  #  [ 'nir_max', 'swir1_max', 'swir2_max', 'EVI']

#    x = get_input_data(['VH_0', 'VV_0', 'VH_2', 'VV_2', 'VH', 'VV', 'slope', 'elevation'], str(2015),
 ##                      ['app_riau', 'app_jambi', 'app_oki', 'Bumitama_PTHungarindoPersada', 'app_kalbar', 'app_kaltim',
    #                    'crgl_stal', 'app_muba'], False)  # ,

  #  ref_study_area = get_reference_raster_from_shape('West_Kalimantan', 'Kalimantan', 2015)
    # x = get_large_area_input_data(ref_study_area, [ 'slope', 'nir_max', 'swir1_max', 'VH_0', 'VV_0', 'VH_2', 'VV_2', 'EVI', 'green_max',
  #  x = get_large_area_input_data(ref_study_area, ['VH_0', 'VV_0', 'VH_2', 'VV_2', 'VH', 'VV', 'slope', 'elevation'],
    #                              'Kalimantan', str(2015), 'West_Kalimantan')



    #ref_study_area = get_reference_raster_from_shape('South_Sumatra', 'Sumatra', 2015)
    # x = get_large_area_input_data(ref_study_area, [ 'slope', 'nir_max', 'swir1_max', 'VH_0', 'VV_0', 'VH_2', 'VV_2', 'EVI', 'green_max',
  #  x = get_large_area_input_data(ref_study_area, ['VH_2', 'VV_2', 'VH', 'VV', 'slope', 'elevation'] ,
       #                          'Sumatra', str(2015), 'South_Sumatra')
#
    #['nir_max', 'swir1_max', 'swir2_max', 'EVI', 'VH_0', 'VV_0']
 ##   ref_study_area = get_reference_raster_from_shape('Riau', 'Sumatra', 2015)
    # x = get_large_area_input_data(ref_study_area, [ 'slope', 'nir_max', 'swir1_max', 'VH_0', 'VV_0', 'VH_2', 'VV_2', 'EVI', 'green_max',
  #  x = get_large_area_input_data(ref_study_area,['VH_2', 'VV_2', 'VH', 'VV', 'slope', 'elevation'],
  #                               'Sumatra', str(2015), 'Riau')

   # x = get_input_data(['VH_0', 'VV_0', 'VH_2', 'VV_2', 'VH', 'VV', 'slope', 'elevation'], str(2015),
    #                   ['app_riau', 'app_jambi', 'app_oki', 'Bumitama_PTHungarindoPersada', 'app_kalbar','app_kaltim', 'crgl_stal', 'app_muba'] , False )#,

    #x = get_input_data(['blue_max', 'green_max', 'red_max', 'nir_max', 'swir1_max', 'swir2_max', 'EVI'], str(2015),['app_kalbar'] , False )
       # , 'swir2_max', 'VH', 'VV','EVI',], 'Kalimantan',  str(2015),  'West_Kalimantan' )
    #x = get_large_area_input_data(ref_study_area,['blue_max', 'nir_max', 'swir1_max', 'swir2_max', 'VH', 'VV','EVI', 'slope'], 'Kalimantan', str(2015),'West_Kalimantan')

# x = get_input_data([ 'blue_max', 'green_max', 'red_max', 'nir_max', 'swir1_max', 'swir2_max', 'VH', 'VV', 'EVI', 'aspect', 'elevation', 'slope'],'Sumatra', str(2015), ['crgl_stal'],False )
    #ref = get_reference_raster_from_shape('app_muba', 'Sumatra')
   # band = get_input_band('swir1_max', 'Sumatra', 2015)
  #  band2 = get_input_band('blue_max', 'Sumatra', 2015)
  #  band3 = get_input_band('swir1_max', 'Sumatra', 2015)
    # for site in sites:
    #     stack_image_input_data(site, bands_base, 'bands_base')
    #     stack_image_input_data(site, bands_radar, 'bands_radar')
    #     stack_image_input_data(site, bands_median, 'bands_median')
    #     #     # stack_image_input_data(site, bands_dem, 'bands_dem')
    #     stack_image_input_data(site, bands_evi2, 'bands_evi2')
    #     stack_image_input_data(site, band_evi2, 'evi2_only')
    #     stack_image_input_data(site, bands_evi2_separate, 'bands_evi2_separate')
    #     stack_image_input_data(site, bands_extended, 'bands_extended')

# trainConcessions = ['app_riau', 'app_jambi']
# get_concession_data(['bands_radar'], trainConcessions)