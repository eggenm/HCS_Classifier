# =============================================================================
# Data_helper.py
# This file does the work of gathering input rasters
# It removes nodata and flattens a stack of raster satellite data
# into a dataframe where each row is a pixel with its n-dimenensional
# observed satellite data.
# =============================================================================


# =============================================================================
# Imports
# =============================================================================
import dirfuncs
import glob
import pandas as pd
import numpy as np
import rasterio.crs
import sklearn.metrics
from sklearn.metrics import f1_score
import ShapefileHelper as shapefilehelp
import hcs_database  as db
from rasterio.mask import mask
import rioxarray as rx
import timer
import imagery_data
import satellite_image_operations as sat_ops

# =============================================================================
# Identify files
# =============================================================================
base_dir = dirfuncs.guess_data_dir()
pixel_window_size = 1
stackData = True
write_input_data = True
image_cache = imagery_data.Imagery_Cache.getInstance()

key_csv = base_dir + 'strata_key.csv'
key_df = pd.read_csv(key_csv)
from_vals = list(key_df['project_code'].astype(float).values)
to_vals = list(key_df['code_3class'].astype(float).values)
to_2class = list(key_df['code_simpl'].astype(float).values)
to_mixedClass = list(key_df['code_mixed_class'].astype(float).values)
three_class_landcoverClassMap = dict( zip(from_vals,to_vals ))
two_class_landcoverClassMap = dict( zip(from_vals,to_2class ))
three_to_two_class_landcoverClassMap = dict( zip(to_vals,to_2class ))
mixed_class_landcoverClassMap = dict( zip(from_vals,to_mixedClass ))

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


########
#get_classes : takes an assessment image and flattens it into a
# dataframe with dimensions/index similar to the input satellite data
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


################################
# combine_input_landcover: merges satellite data with observed landcover
###############################
def combine_input_landcover(input, landcover_all):
    try:
        with timer.Timer() as t:
            data_df = landcover_all.merge(input, left_index=True, right_index=True, how='left')
            data_df[data_df <= -999] = np.nan  #MEE 6-1-2020:  This used to be conditional on whether I was using the data in a training set or not. If I was doing it to make a class map then I did not do this

            return data_df
    finally:
        print('combine_input_landcover Request took %.03f sec.' % t.interval)




def get_input_band(band, name, year):
    try:
        with timer.Timer() as t:
            print(band)
            image_cache = imagery_data.Imagery_Cache.getInstance()
            print('IMAGE CACHE SINGLETON: ',image_cache)
            if(db.data_context_dict!='supplementary_class'):
                image = image_cache.get_band_by_context_year(band, name, year)
    finally:
        print('Get' , band , ' Request took %.03f sec.' % t.interval)
    return image


def trim_input_band_by_shape(input_raster, boundary):
    out_img, out_transform = mask(input_raster, shapes=boundary, crop=True)
    return out_img, out_transform


def reproject_match_input_band(band, island, year, bounding_raster):
    image2 = bounding_raster # rx.open_rasterio(bounding_raster)
    print('image2.shape:  ', image2.shape)
    image3 = get_input_band(band, island, year)
    print('image3.shape:  ', image3.shape)
    if(image3.dtype=='float64'):
        image3.data  = np.float32(image3)
    image3 = image3.rio.reproject_match(image2)
    print('image2.shape:  ', image2.shape)
    print('image3.shape:  ', image3.shape)

    image2=False
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
                 block = data_src[window.col_off:window.col_off+window.width,window.row_off:window.row_off+window.height ]#.read(window=window)
                 dst.write(block , window=window)
        dst.close()




def get_feature_inputs(band_groups, bounding_box,  year, concession=None, filename=None):
    tif=''
    try:
        with timer.Timer() as t:
            print('Band_Groups:  ',band_groups)
            x = range(0, bounding_box.shape[1])
            y = range(0, bounding_box.shape[2])
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
                        print("CONTEXT SUPPLEMENTAL CLASS  file: ", filename, "  BAND:  ", band)
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
                out_img = np.asarray(out_img[0]).flatten()
                if windows.empty :
                    windows = pd.DataFrame({i: out_img}, index=full_index)
                else:
                    windows = windows.merge(pd.DataFrame({i: out_img}, index=full_index), left_index=True, right_index=True, how='left')

                out_img = False
            windows.index.names = ['i', 'j']
            full_index = False
            return windows
    finally:
        print('get_feature_inputs took %.03f sec.' % t.interval)



def get_concession_bands(bands, year, bounding_box, concession=None, filename = None):
    try:
        x=False
        with timer.Timer() as t:
            return(get_feature_inputs(bands, bounding_box,  year, concession, filename))

    finally:
        print('get_concession_bands Request took %.03f sec.' % t.interval)
    #return x



def get_fixed_bands(bands, name, context, year='NONE' ):
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
                    data = get_fixed_bands(bands, site, type)

                else:
                    island = db.data_context_dict[site]
                    all_class = image_cache.get_class_by_concession_name(site)
                    #box = shapefilehelp.get_bounding_box_polygon(db.shapefiles[site])
                    if not get_predictor_data_only:
                        #THIS IS TRAINING DATA
                        year = str(int(db.get_concession_assessment_year(site)))
                        x = get_concession_bands(bands, year, all_class, site)
                        y = get_classes(all_class.data, 'clas')
                        data = combine_input_landcover(x, y)
                    elif get_predictor_data_only:
                        data = get_concession_bands(bands, year, all_class, site)
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
                print( "*****dataHalper x.shape:  " , x.shape)
                x_in = x.shape[0]
                x = fill_no_data(x)
                x_out = x.shape[0]
                print("*****dataHalper x.shape after drop no data:  ", x.shape)
                if x_in!=x_out:
                    raise RuntimeError
                return x
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
    print(three_class_landcoverClassMap)
    return pd.Series(X).map(three_class_landcoverClassMap)

def map_to_2class(X):
    return pd.Series(X).map(two_class_landcoverClassMap)

def map_3_to_2class(X):
    return pd.Series(X).map(three_to_two_class_landcoverClassMap)

def map_to_mixed_classes(X):
    return pd.Series(X).map(mixed_class_landcoverClassMap)

def trim_data(input):
    return input.groupby('clas').filter(lambda x: len(x) > 10000)

def trim_data2(input):
    #return input[input.clas.isin([21.0, 18.0, 7.0, 6.0, 4.0, 5.0, 20.0])]
    return input[np.logical_not(input.clas.isin([8.0, 13.0, 14.0, 18.0]))] #enclosure, mixed coconut, mixed rubber, smallholder mixed


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
        #fill = 0
        fill = np.nan
        with timer.Timer() as t:
            data[data <= -999] = fill
            data[data == 255] = fill
            data[data >= 999] = fill

            data.dropna(inplace=True)
            return data
    finally:
        print('Drop NoData Request took %.03f sec.' % t.interval)

def fill_no_data(data):
    try:
        #fill = 0
        fill = np.nan
        with timer.Timer() as t:
            data[data <= -999] = fill
            data[data == 255] = fill
            data[data >= 999] = fill

            data = data.fillna(value=100)
            return data
    finally:
        print('Drop NoData Request took %.03f sec.' % t.interval)



def do_comparison(concession_name):
    print("comparing:  ", concession_name)
    dirname = 'C:\\Users\\ME\\Dropbox\\HCSA_GIS\\' + concession_name +'\\'
    assessment_file = dirname + concession_name+ '_all_class.remapped.tif'
    class_file = dirname + concession_name+ 'FINAL_classified_by_ensemble_rf.tif'
    file = glob.glob(assessment_file)
    assessment = rx.open_rasterio(file[0])
    file = glob.glob(class_file)
    predict_data = rx.open_rasterio(file[0])
    y = get_classes(assessment.data, 'clas')
    y = y.iloc[:, 0]
    y_hat =  get_classes(predict_data.data, 'clas')
    y_hat = y_hat.iloc[:, 0]
    y_mixed = map_to_mixed_classes(y)
    y_2class = map_to_2class(y)
    df = y_mixed.to_frame(name='mixed').join(y_2class.to_frame(name='2class')).join(y_hat.to_frame(name='y_hat'))
    print("BEFORE drop no data:  ", df.size)
    df = drop_no_data(df)
    print("AFTER drop no data:  ", df.size)
    # train_df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    df = df[indices_to_keep]
    print("AFTER intrmediate drop:  ", df.size)
    show_results(df.loc[ : , 'mixed' ], df.loc[ : , 'y_hat' ])
    show_results(df.loc[ : , '2class' ], df.loc[ : , 'y_hat' ])
    indices_to_keep = ~df.isin([8, 13, 14, 18, 8.0, 13.0, 14.0, 18.0]).any(1)
    df = df[indices_to_keep]
    print("AFTER LAST drop:  ", df.size)
    show_results(df.loc[:, 'mixed'], df.loc[:, 'y_hat'])
    print('END_COMPARISON')


if __name__ == "__main__":
    ################################################
    #This runnable section is a convenient way to prep input data
    # for different concessions, where you have a rasterized asssessment
    # or any area for which you have a shapefile.
    # #############################################

    #When you want to prep all concessions used as training
    # The method get_input_data will try to find the input data in the filesystem
    # and will write (as geotiff) it to the file system if not found
    concessions_csv = base_dir + 'concession_inventory.csv'
    con_df = pd.read_csv(concessions_csv)
    my_dict = sat_ops.s2_band_dict.copy()
    my_dict.update(sat_ops.s1_band_dict)
    my_dict.update(sat_ops.dem_band_dict)
    bands = my_dict.values()
    for index, row in con_df.iterrows():
        print(row['app_key'], bool(row['ingest']), row['assessment_year'])
        if(bool(row['ingest'])):
            x = get_input_data(bands, str(int(row['assessment_year'])), [row['app_key']], False)
    ##########################################################################

    #########################################################################
    #This is an example where you might only have a shapefile to start
    bands = [   'blue_max',
        'red_max',
        'nir_max',
        'swir1_max','EVI', 'swir2_max',
          'brightness'     , 'greenness' , 'wetness',
        'VH_2', 'VV_2',
         'VH', 'VV', 'VH_0', 'VV_0', 'VH_2', 'VV_2',  'slope'
    ]
    ref_study_area = get_reference_raster_from_shape('gunung_palung', 'Kalimantan', 2017)
    x = get_large_area_input_data(ref_study_area, bands,
                                  'Kalimantan', str(2017), 'gunung_palung')
    #########################################################
    #This will show accuracy for a concession classification vs its HCS assessmnt
    do_comparison('mukti_prakarsa')