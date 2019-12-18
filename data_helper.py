# =============================================================================
# Imports
# =============================================================================
import dirfuncs
import os
import re
import glob
import pandas as pd
import numpy as np
import rasterio as rio
import rasterio.crs
from sklearn.preprocessing import StandardScaler, LabelEncoder


# =============================================================================
# Identify files
# =============================================================================
base_dir = dirfuncs.guess_data_dir()
pixel_window_size = 1
stackData = True

#classes = {1: "HCSA",
     #      0: "NA"}
# sites = ['gar_pgm',
#     'app_riau',
#   'app_kalbar',
#          'app_kaltim',
#       'app_jambi',
#  'app_oki',
#         'crgl_stal'
#     ]

bands_base=['S2_blue_max', 'S2_green_max', 'S2_red_max', 'S2_nir_max', 'S2_nir2_max', 'S2_swir1_max', 'S2_swir2_max', 'S2_swir3_max', 'S2_vape_max']

bands_historical=['ndvi_2013', 'ndvi_2014', 'ndvi_2015', 'ndvi_2011', 'ndvi_2010', 'ndvi_2009', 'ndvi_2008', 'ndvi_2007', 'ndvi_2006', 'ndvi_2005', 'ndvi_2004', 'ndvi_2003']

bands_median=['S2_blue_median', 'S2_green_median', 'S2_red_median', 'S2_nir_median', 'S2_nir2_median', 'S2_swir1_median', 'S2_swir2_median', 'S2_swir3_median', 'S2_vape_median']

bands_radar=['VH_2015', 'VV_2015']

bands_dem=['elevation', 'slope']

bands_extended=['rededge1', 'rededge2', 'rededge3']

bands_evi2_separate=['S2_red_max', 'S2_nir_max']

band_evi2 = ['EVI2_s2_max']

bands_evi2 = ['S2_red_max', 'S2_nir_max', 'EVI2_s2_max']

key_csv = '/Users/ME/Dropbox/HCSproject/data/strata_key.csv'
key_df = pd.read_csv(key_csv)
from_vals = list(key_df['project_code'].astype(float).values)
to_vals = list(key_df['code_3class'].astype(float).values)
landcoverClassMap = dict( zip(from_vals,to_vals ))



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
    shape = array.shape
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
    return (windows)


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
    #three_class_file = base_dir + concession + '/' + concession + '_remap_3class.remapped.tif'
    allclass_file = base_dir + concession + '/' + concession + '_all_class.remapped.tif'
    #print(three_class_file)
    #file_list = sorted(glob.glob(three_class_file))
    ## Read classification labels
    #with rio.open(file_list[0]) as clas_src:
        #three_class = clas_src.read()
    file_list = sorted(glob.glob(allclass_file))
    with rio.open(file_list[0]) as clas_src:
        all_class = clas_src.read()
    return  all_class


def get_classes(classImage, name):
    clas_dict = {}
    shape = classImage.shape
    for i in range(classImage.shape[1]):
        for j in range(classImage.shape[2]):
            clas_dict[(i, j)] = classImage[0, i, j]
    full_index = pd.MultiIndex.from_product([range(shape[1]), range(shape[2])], names=['i', 'j'])
    classes = pd.DataFrame({name: pd.Series(clas_dict)}, index=full_index)
    return classes


def combine_input_landcover(input, landcover_all):
    data_df = landcover_all.merge(input, left_index=True, right_index=True, how='left')
    #data_df = landcover2.merge(data_df, left_index=True, right_index=True, how='left')
    data_df[data_df <= -999] = np.nan
    data_df = data_df.dropna()
    #print('*****data_df shape:  ', data_df.shape)
    return data_df


def scale_data(x):
    # print('x min:  ', x.min())
    # print('mean ', x.mean())
    # print('xmax:  ', x.max())
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x.astype(np.float64))
    # print('x_scaled min:  ', x_scaled.min())
    # print('x scaled mean ', x.mean())
    # print('x_scaled max:  ', x_scaled.max())
    return x_scaled


def mask_water(an_img, concession):
    with rio.open(base_dir + concession + "/in/" + concession + "_radar.VH_2015.tif") as radar1:
        radar = radar1.read()
    watermask = np.empty(radar.shape, dtype=rasterio.uint8)
    watermask = np.where(radar > -17.85, 1, -999999).reshape(radar.shape[1], radar.shape[2])
    an_img = an_img * watermask
    return an_img

def get_feature_inputs(band_groups, concession):
    srcs_to_mosaic=[]
    outtif=''
    for bands in band_groups:
        outtif = os.path.join(base_dir, concession, 'out', 'input_' + concession + '_' + bands + '.tif')
        #print(outtif)
        file = glob.glob(outtif)
        srcs_to_mosaic.append(file[0])
        #print(srcs_to_mosaic)
    array = []
    for ii, ifile in enumerate(srcs_to_mosaic):
        bands = rio.open(srcs_to_mosaic[ii]).read()
        if bands.shape[0] > 1:
            for i in range(0, bands.shape[0]):
                band=bands[i]
                array.append(band)
        elif bands.shape[0] == 1:
            band = np.squeeze(bands)
            array.append(band)
    return array


def get_concession_bands(bands, concession):
    img = get_feature_inputs(bands, concession)
    array = np.asarray(img)
    x = gen_windows(array, pixel_window_size)
    return x

def get_concession_data(bands, concessions):
    data = pd.DataFrame()
    if(isinstance(concessions, str)):
        all_class_image = get_landcover_class_image(concessions)
        # class_image = mask_water(class_image, concession)
        y = get_classes(all_class_image, 'clas')
        #y2 = get_classes(two_class_image, 'class_remap')
        x = get_concession_bands(bands, concessions)
        data = combine_input_landcover(x, y)
    else:
        for concession in concessions:
            all_class_image = get_landcover_class_image(concession)
            # class_image = mask_water(class_image, concession)
            y = get_classes(all_class_image, 'clas')
            #y2 = get_classes(two_class_image, 'class_remap')
            x = get_concession_bands(bands, concession)
            if data.empty:
                data = combine_input_landcover(x, y)
            else:
                data = pd.concat([data, combine_input_landcover(x, y)], ignore_index=True)
    return data

def get_all_concession_data(concessions):
    data = pd.DataFrame()
    for concession in concessions:
        outtif = base_dir + concession + '/out/input_' + concession +'.tif'
        if(stackData):
            outtif, bands = stack_image_input_data(concession)

        with rio.open(outtif) as img_src:
            img = img_src.read()
            x = gen_windows(img, pixel_window_size)
            #print('x.shape:  ', x.shape)
            x.columns=bands
        all_class_image = get_landcover_class_image(concession)
        # class_image = mask_water(class_image, concession)
        y = get_classes(all_class_image, 'clas')
        #y2 = get_classes(two_class_image, 'class_remap')
       # print('y.shape:  ', y.shape)
        if data.empty:
            data = combine_input_landcover(x, y)
        else:
            data = pd.concat([data, combine_input_landcover(x, y)], ignore_index=True)
           # print("  data.shape:  ", data.shape)
    return data


def remove_low_occurance_classes( X, class_data):
    df= pd.DataFrame(data=[X, class_data])
    threshold = 10  # Anything that occurs less than this will be removed.
    df = df.groupby('clas').filter(lambda x: len(x) > threshold)

def map_to_3class(X):
    return pd.Series(X).map(landcoverClassMap)

#print(landcoverClassMap)
#for site in sites:
    # stack_image_input_data(site, bands_base, 'bands_base')
    #     # stack_image_input_data(site, bands_radar, 'bands_radar')
    #     # stack_image_input_data(site, bands_median, 'bands_median')
    #     # stack_image_input_data(site, bands_dem, 'bands_dem')
    # stack_image_input_data(site, bands_evi2, 'bands_evi2')
    # stack_image_input_data(site, band_evi2, 'evi2_only')
    # stack_image_input_data(site, bands_evi2_separate, 'bands_evi2_separate')
    #stack_image_input_data(site, bands_extended, 'bands_extended')

# trainConcessions = ['app_riau', 'app_jambi']
# get_concession_data(['bands_radar'], trainConcessions)