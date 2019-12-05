import matplotlib.pyplot as plt
from numpy.distutils.misc_util import as_list

import data_helper as helper
import dirfuncs
import rasterio as rio
from rasterio.rio.stack import stack
import hcs_database as hcs_db
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import os
from sklearn.model_selection import train_test_split, cross_val_score, \
    GridSearchCV, cross_val_predict, ShuffleSplit, learning_curve, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier as rfc

#############   SETUP  PARAMS    ######################
classConcession = 'app_riau'
base_dir = dirfuncs.guess_data_dir()
band_set ={1:'bands_radar', 2: 'bands_base', 3: 'bands_median', 4: ['bands_base','bands_radar'], 5: ['bands_base','bands_radar','bands_dem'], 6: ['bands_radar','bands_dem']}
raster_out_path = os.path.join(base_dir, classConcession, 'out', "raster.tiff")
pixel_window_size=1
doGridSearch=True

#def buildModel():

def get_feature_inputs(band_groups):
    srcs_to_mosaic=[]
    outtif=''
    print(band_groups)
    if(isinstance(band_groups, str)):
        outtif = os.path.join(base_dir, classConcession, 'out', 'input_' + classConcession + '_' + band_groups + '.tif')
        print(outtif)
        with rio.open(outtif) as img_src:
            array = img_src.read()
    else:
        for bands in band_groups:
            outtif = os.path.join(base_dir, classConcession, 'out', 'input_' + classConcession + '_' + bands + '.tif')
            print(outtif)
            file = glob.glob(outtif)
            srcs_to_mosaic.append(file[0])
            print(srcs_to_mosaic)
        #array, raster_prof = es.stack(srcs_to_mosaic)#, out_path=raster_out_path)
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
        #print(array.shape)
        print(array[:15])
    return array

def run_model(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.0040, test_size=0.1,
                                                        random_state=13)
    print(pd.Series(y_train).value_counts())
    clf = rfc(n_estimators=40, max_depth=6, max_features=.3, max_leaf_nodes=10,
              random_state=13, oob_score=True, n_jobs=-1,
              #  class_weight = {0:0.33, 1: 0.33, 2: 0.34})
              class_weight='balanced')
    if doGridSearch:
        param_grid = [{'max_depth': [2, 6, 10],
                       'max_leaf_nodes': [10, 20, 50],
                       'max_features': [.25, .5, .75 ],
                       'n_estimators': [20, 100, 250, 500]}]
        grid_search = GridSearchCV(clf, param_grid, cv = 5, #scoring = 'balanced_accuracy',
                                   return_train_score = True, refit = True)

        grid_search.fit(X_train, y_train)

        randomforest_fitted_clf = grid_search.best_estimator_
    else:
        randomforest_fitted_clf = clf.fit(X_train, y_train)
    print(randomforest_fitted_clf.get_params())
    return randomforest_fitted_clf.score(X_test, y_test)

def evaluate_model():
    df = pd.DataFrame(index=band_set.keys(), columns=['score_all', 'score_binary'])
    for key, bands in band_set.items():
        img=get_feature_inputs(bands)
        array = np.asarray(img)
        x = helper.gen_windows(array, pixel_window_size)
        two_class_y, all_class_y = helper.get_landcover_class_image(classConcession)
        y = helper.get_classes(all_class_y, 'clas')
        y2 = helper.get_classes(two_class_y, 'class_binary')
        data = helper.combine_input_landcover(x, y, y2)
        data = data.groupby('clas').filter(lambda x: len(x) > 1000)
        X = data[[col for col in data.columns if ((col != 'class') & (col != 'class_binary'))]]
        X_scaled = helper.scale_data(X)
        landcover = data['clas'].values
        score_all = run_model(X_scaled, landcover)
        landcover = data['class_binary'].values
        score_binary = run_model(X_scaled, landcover)
        scores = [score_all, score_binary]
        df.loc[key] = scores
        print(df)
    print(df)

evaluate_model()
# img=get_feature_inputs(band_set.get(5))
# array=np.asarray(img)
# x = helper.gen_windows(array, pixel_window_size)
# two_class_y, all_class_y = helper.get_landcover_class_image(classConcession)
# y = helper.get_classes(all_class_y, 'clas')
# y2 = helper.get_classes(two_class_y, 'class_binary')
# data = helper.combine_input_landcover(x, y, y2)
# data = data.groupby('clas').filter(lambda x: len(x) > 1000)
# X = data[[col for col in data.columns if ((col != 'class') & (col != 'class_binary'))]]
# X_scaled = helper.scale_data(X)
