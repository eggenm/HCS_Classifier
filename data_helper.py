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

# =============================================================================
# Identify files
# =============================================================================
base_dir = dirfuncs.guess_data_dir()
pixel_window_size = 1
stackData = True
write_input_data = True


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

class imagery_cache:

    def __init__(self):
        self.island_data_table = {}

    def get_band_by_island_year(self, band, island, year ):
        key = island+str(year)+band
        try:
            self.island_data_table[key]
        except KeyError:
            tif = os.path.join(base_dir, island, 'out', str(year), '*' + band + '.tif')
            file = glob.glob(tif)
            self.island_data_table[key] = rx.open_rasterio(file[0])
        return self.island_data_table[key]

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
            tuples = list(it.product(x, y))
            myfunc = lambda a: a.flatten()
            aList = [myfunc(array[i, :, :]) for i in range(0, array.shape[0])]
            full_index = pd.MultiIndex.from_product([x, y], names=['i', 'j'])
            i = 0
            windows = pd.DataFrame({i: aList[0]}, index=full_index)
            for i in range(1, len(aList)):
                temp = pd.DataFrame({i: aList[i]}, index=full_index)
                windows = windows.merge(temp, left_index=True, right_index=True, how='left')
            windows.index.names = ['i', 'j']
            aList=False
    finally:
        print('gen_windows2 request took %.03f sec.' % t.interval)
    return windows


def stack_image_input_data(concession, bands, name):
    input_dir = base_dir + concession + "/in/"
    #print(input_dir)
    outtif = base_dir + concession + '/out/input_' + concession + '_'+ name + '.tif'
    if stackData:
        #print(input_dir + "*" + bands[0] + "*.tif")
        file_list = sorted(glob.glob(input_dir + "*" + bands[0] + "*.tif"))
        with rasterio.open(file_list[0]) as src0:
            meta = src0.meta

        # Update meta to reflect the number of layers
        meta.update(count=len(bands), dtype='float64')

        # Read each layer and write it to stack
        with rasterio.open(outtif, 'w', **meta) as dst:
            for i, band in enumerate(bands, start=1):
               # print(i, '....', band)
                layer = sorted(glob.glob(input_dir + "*" + band + "*.tif"))[0]
                name = os.path.basename(layer)
                if re.search('median', name)is not None and re.search('median', name).span()[0]>0:
                        name = 'median_'+name.split('.', 3)[1]
                else:
                    name = name.split('.', 3)[1]
                with rasterio.open(layer) as src1:
                    band = src1.read(1).astype('float64')
                    # print('Max:  ', band.max())
                    # print('Min:  ', band.min())
                    dst.write_band(i, band)
        dst.close()
    return outtif, bands


def get_landcover_class_image(concession):
    print(concession)
    #three_class_file = base_dir + concession + '/' + concession + '_remap_3class.remapped.tif'
    allclass_file = base_dir + concession + '/' + concession + '_all_class.remapped.tif'
    print("**get_landcover_class_image:  allclass_file:  ", allclass_file)
    return allclass_file


def get_classes2(classImage, name):
    try:
        with timer.Timer() as t:
            x = range(0, classImage.shape[1])
            y = range(0, classImage.shape[2])
            full_index = pd.MultiIndex.from_product([x, y], names=['i', 'j'])
            myfunc = lambda a: a.flatten()
            aList = [myfunc(classImage[i, :, :]) for i in range(0, classImage.shape[0])]
            classes = pd.DataFrame({name: aList[0]}, index=full_index)
    finally:
        print('get_classes2 Request took %.03f sec.' % t.interval)
    return classes


def get_classes(classImage, name):
    clas_dict = {}
    shape = classImage.shape
    for i in range(classImage.shape[1]):
        for j in range(classImage.shape[2]):
            clas_dict[(i, j)] = classImage[0, i, j]
    full_index = pd.MultiIndex.from_product([range(shape[1]), range(shape[2])], names=['i', 'j'])
    classes = pd.DataFrame({name: pd.Series(clas_dict)}, index=full_index)
    return classes


def combine_input_landcover(input, landcover_all, isClass):
    data_df = landcover_all.merge(input, left_index=True, right_index=True, how='left')
    #data_df = landcover2.merge(data_df, left_index=True, right_index=True, how='left')
    if( not isClass):
        data_df[data_df <= -999] = np.nan
        data_df = data_df.dropna()
    #print('*****data_df shape:  ', data_df.shape)
    return data_df


def scale_data(x):
    try:
        with timer.Timer() as t:
            print('DEPRECATED - SCALING DONE ON INGEST')
            #scaler = StandardScaler()
            #x_scaled = scaler.fit_transform(x.astype(np.float64))
            #return x_scaled
    finally:
        print('ScaleData Request took %.03f sec.' % t.interval)


def mask_water(an_img, concession):
    with rio.open(base_dir + concession + "/in/" + concession + "_radar.VH_2015.tif") as radar1:
        radar = radar1.read()
    watermask = np.empty(radar.shape, dtype=rasterio.uint8)
    watermask = np.where(radar > -17.85, 1, -999999).reshape(radar.shape[1], radar.shape[2])
    an_img = an_img * watermask
    return an_img

def get_input_band(band, name, year):
    try:
        with timer.Timer() as t:
            print(band)
            image_cache = imagery_data.Imagery_Cache.getInstance()
            print('IMAGE CACHE SINGLETON: ',image_cache)
            image = image_cache.get_band_by_island_year(band, name, year)
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
            for ji, window in dst.block_windows(1):  # or ref file here
                 print('ji:  ', ji)
                 print('window:  ', window)
               #  print('window.shape:  ', window.shape)
                 block = data_src[window.col_off:window.col_off+window.width,window.row_off:window.row_off+window.height ]#.read(window=window)
            #     if sum(sum(sum(~np.isnan(block)))) > 0:
                 dst.write(block , window=window)
        dst.close()



def get_feature_inputs(band_groups, bounding_box, island, year, concession=None):
    srcs_to_mosaic=[]
    tif=''
    print('Band_Groups:  ',band_groups)
    #all_bands = list(itertools.chain(*band_groups))
    #all_bands = band_groups.flatt
    #print('ALL_BANDS:', all_bands)
    array = [0 for x in range(len(band_groups))]
    print('len(array):  ', len(array))
    for i, band in enumerate(band_groups):
        if(concession):
            tif = base_dir + concession + '/out/' +year+ '/input_' + concession +'_'+ band + '.tif'
            try:
                file_list = sorted(glob.glob(tif))
                out_img =  rx.open_rasterio(file_list[0])
            except:
                print('except: ', band , concession, island)
                out_img = reproject_match_input_band(band, island, year, bounding_box)
                if (write_input_data):
                    write_concession_band(out_img, bounding_box,  tif)
        else:
            out_img = reproject_match_input_band(band, island, year, bounding_box)


        #out_img = trim_input_band_by_raster(file[0], bounding_box, band)

            # write_data_array(file, 'app_oki', band, bounding_box)
        #srcs_to_mosaic.append(out_img)
        #print(srcs_to_mosaic)


    ##for ii, ifile in enumerate(srcs_to_mosaic):
    #    bands = rio.open(srcs_to_mosaic[ii]).read()
    #     if out_img.shape[0] > 1:
    #         for i in range(0, out_img.shape[0]):
    #             band=out_img[i]
    #             array.append(band)
    #     elif out_img.shape[0] == 1:
    #         band = np.squeeze(out_img)
        print('I:  ',i)
        array[i] = np.asarray(out_img[0])
        out_img = False


    #array = rxmerge.merge_arrays(array)
    return np.asarray(array)


def write_data_array(file, concession, band, boundary):
    with rasterio.open(file[0]) as image:
        meta = image.meta
        crs = image.crs
        #dtype = rio.float64
        img, trans = trim_input_band_by_shape(image, boundary)
        print('TRANSFORM: ', trans)
    print('META:  ',meta)
    print('SHAPE:  ',img.shape)
    img = np.squeeze(img)
    print('META_AFTER_UPDATE:  ', meta)
    # Update meta to reflect the number of layers
    outtif = base_dir + concession + '/out/input_' + concession + band  + '.tif'
    with rasterio.open(outtif, 'w', driver = 'GTiff',
     #this is wrong, i think height and width mixed up             height = img.shape[0], width = img.shape[1],
                  crs = crs, dtype = img.dtype,
                  count = 1, transform = trans) as dst:
                dst.write_band(1, img)
    dst.close()
    print('TEST')

def get_concession_bands(bands, island, year, bounding_box, concession=None):
    try:
        x=False
        with timer.Timer() as t:
            img = get_feature_inputs(bands, bounding_box, island, year, concession)
            #array = np.asarray(img[0])
            x = gen_windows2(img)
            #x = gen_windows(img, pixel_window_size)
    finally:
        array=False
        img=False
        print('get_concession_bands Request took %.03f sec.' % t.interval)
    return x


def get_input_data(bands, year, concessions, isClass=False):
    data = pd.DataFrame()
    for concession in concessions:
        print(concession)
        island = db.conncession_island_dict[concession]
        all_class_image = get_landcover_class_image(concession)
        print(all_class_image)
        class_file = sorted(glob.glob(all_class_image))[0]
        # if(write_input_data):
        #    write_data_array(file_list[0],concession,'class',)

        all_class = rx.open_rasterio(class_file)
        if(write_input_data):
            print('TODO - check class image, shape: ', all_class.shape)
           # write_data_array(class_file, 'Class'+concession)
        y = get_classes(all_class.data, 'clas')
        #box = shapefilehelp.get_bounding_box_polygon(db.shapefiles[concession])
        x = get_concession_bands(bands, island, year, all_class, concession)
        if data.empty:
            data = combine_input_landcover(x, y, isClass)
        else:
            data = pd.concat([data, combine_input_landcover(x, y, isClass)], ignore_index=True)
    all_class=False
    x=False
    y=False
    return data

def get_large_area_input_data(study_area_base_raster, bands, island, year, name=None):
        try:
            with timer.Timer() as t:
                x = get_concession_bands(bands, island, year, study_area_base_raster, name)
                x = drop_no_data(x)
                #X_scaled_class = scale_data(x)
                return x
               # print('X_scaled_class.shape:  ', X_scaled_class.shape)
        finally:
            x = False
            print('Get Input Data Request took %.03f sec.' % t.interval)


def get_reference_raster_from_shape(shapefile, island, year):
    bounding = shapefilehelp.get_bounding_box_polygon(db.shapefiles[shapefile])
    outtif = get_input_band('blue_max', island, year)
    #out_img = reproject_match_input_band(outtif)
    out_img =outtif.rio.clip(bounding, outtif.rio.crs)
    outtif = False
    return out_img


# def get_concession_data(bands, concessions, isClass=False):
#     data = pd.DataFrame()
#     if(isinstance(concessions, str)):
#         all_class_image = get_landcover_class_image(concessions)
#         # class_image = mask_water(class_image, concession)
#         y = get_classes(all_class_image, 'clas')
#         #y2 = get_classes(two_class_image, 'class_remap')
#         x = get_concession_bands(bands, concessions)
#         data = combine_input_landcover(x, y, isClass)
#     else:
#         for concession in concessions:
#             all_class_image = get_landcover_class_image(concession)
#             # class_image = mask_water(class_image, concession)
#             y = get_classes(all_class_image, 'clas')
#             #y2 = get_classes(two_class_image, 'class_remap')
#             x = get_concession_bands(bands, concession)
#             if data.empty:
#                 data = combine_input_landcover(x, y, isClass)
#             else:
#                 data = pd.concat([data, combine_input_landcover(x, y, isClass)], ignore_index=True)
#     return data
#
# def get_all_concession_data(concessions, isClass=False):
#     data = pd.DataFrame()
#     for concession in concessions:
#         outtif = base_dir + concession + '/out/input_' + concession +'.tif'
#         if(stackData):
#             outtif, bands = stack_image_input_data(concession)
#
#         with rio.open(outtif) as img_src:
#             img = img_src.read()
#             x = gen_windows(img, pixel_window_size)
#             #print('x.shape:  ', x.shape)
#             x.columns=bands
#         all_class_image = get_landcover_class_image(concession)
#         # class_image = mask_water(class_image, concession)
#         y = get_classes(all_class_image, 'clas')
#         #y2 = get_classes(two_class_image, 'class_remap')
#        # print('y.shape:  ', y.shape)
#         if data.empty:
#             data = combine_input_landcover(x, y, isClass)
#         else:
#             data = pd.concat([data, combine_input_landcover(x, y, isClass)], ignore_index=True)
#            # print("  data.shape:  ", data.shape)
#     return data


def remove_low_occurance_classes( X, class_data):
    df= pd.DataFrame(data=[X, class_data])
    threshold = 10  # Anything that occurs less than this will be removed.
    df = df.groupby('clas').filter(lambda x: len(x) > threshold)

def map_to_3class(X):
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
            return data.dropna()
    finally:
        print('Drop NoData Request took %.03f sec.' % t.interval)

input_data_cache = imagery_cache()
#print(landcoverClassMap)
if __name__ == "__main__":
    #write_input_data=True
    #x = get_input_data(['aspect', 'VH', 'blue_max', 'EVI'],'Kalimantan', str(2015), ['app_kalbar'], False)
    x = get_input_data(['VH_0', 'VV_0', 'VH_2', 'VV_2', 'VH', 'VV', 'slope', 'elevation'],  str(2015), ['gar_pgm', 'Bumitama_PTGemilangMakmurSubur','PTAgroAndalan','PTMitraNusaSarana', 'Bumitama_PTDamaiAgroSejahtera']
                       , False )#,

    x = get_input_data(['VH_0', 'VV_0', 'VH_2', 'VV_2', 'VH', 'VV', 'slope', 'elevation'], str(2015),
                       ['app_riau', 'app_jambi', 'app_oki', 'Bumitama_PTHungarindoPersada', 'app_kalbar', 'app_kaltim',
                        'crgl_stal', 'app_muba'], False)  # ,

    ref_study_area = get_reference_raster_from_shape('West_Kalimantan', 'Kalimantan', 2015)
    # x = get_large_area_input_data(ref_study_area, [ 'slope', 'nir_max', 'swir1_max', 'VH_0', 'VV_0', 'VH_2', 'VV_2', 'EVI', 'green_max',
  #  x = get_large_area_input_data(ref_study_area, ['VH_0', 'VV_0', 'VH_2', 'VV_2', 'VH', 'VV', 'slope', 'elevation'],
    #                              'Kalimantan', str(2015), 'West_Kalimantan')



    ref_study_area = get_reference_raster_from_shape('Jambi', 'Sumatra', 2015)
    # x = get_large_area_input_data(ref_study_area, [ 'slope', 'nir_max', 'swir1_max', 'VH_0', 'VV_0', 'VH_2', 'VV_2', 'EVI', 'green_max',
  #  x = get_large_area_input_data(ref_study_area, ['VH_0', 'VV_0', 'VH_2', 'VV_2', 'VH', 'VV', 'slope', 'elevation'],
            #                     'Sumatra', str(2015), 'Jambi')

   # x = get_input_data(['VH_0', 'VV_0', 'VH_2', 'VV_2', 'VH', 'VV', 'slope', 'elevation'], str(2015),
    #                   ['app_riau', 'app_jambi', 'app_oki', 'Bumitama_PTHungarindoPersada', 'app_kalbar','app_kaltim', 'crgl_stal', 'app_muba'] , False )#,

    x = get_input_data(['blue_max', 'green_max', 'red_max', 'nir_max', 'swir1_max', 'swir2_max', 'EVI'], str(2015),['app_kalbar'] , False )
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