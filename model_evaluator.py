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
import sklearn.metrics
from sklearn.metrics import f1_score

#############   SETUP  PARAMS    ######################
sites = [#'gar_pgm',
    'app_riau',
  'app_kalbar',
         'app_kaltim',
      'app_jambi',
 'app_oki',
       # 'crgl_stal'
    ]
base_dir = dirfuncs.guess_data_dir()
band_set ={#1:['bands_radar'],
           2: ['bands_base'],
           3: ['bands_median'],
           4: ['bands_base','bands_radar'],
       #    5: ['bands_base','bands_radar','bands_dem']#,
      #     6: ['bands_radar','bands_dem']
           7:['bands_evi2_separate'],
           8:['evi2_only'],
           #9:['bands_evi2'],
           10:['bands_base','bands_radar','evi2_only'],
           11:['bands_base','bands_median','bands_radar','evi2_only']
           }

pixel_window_size=1
doGridSearch=True

#def buildModel():

def show_results(y_test, y_hat):
    report = sklearn.metrics.classification_report(y_test, y_hat, output_dict=True)
    print(report)
    # export_report = pd.DataFrame(report).to_csv(r'C:\Users\ME\Desktop\export_report_riau.csv', index=None,
    #                           header=True)
    confMatrix = sklearn.metrics.confusion_matrix(y_test, y_hat)
    print(confMatrix)


def train_model(X_train, y_train):
    #print('Training:  ', pd.Series(y_train).value_counts())
    #print(X_train[:7])
    clf = rfc(n_estimators=40, max_depth=6, max_features=.3, max_leaf_nodes=10,
              random_state=16, oob_score=True, n_jobs=-1,
              #  class_weight = {0:0.33, 1: 0.33, 2: 0.34})
              class_weight='balanced')
    if doGridSearch:
        print(" ############  IN GRID SEARCH  ############# ")
        param_grid = [{'max_depth': [6, 10, 16],
                       'max_leaf_nodes': [40, 60, 100],
                       'max_features': [.25, .5, .75 ],
                       'n_estimators': [100, 250, 500]}]
        grid_search = GridSearchCV(clf, param_grid, cv = 5, #scoring = 'balanced_accuracy',
                                   return_train_score = True, refit = True)

        grid_search.fit(X_train, y_train)

        randomforest_fitted_clf = grid_search.best_estimator_
    else:
        randomforest_fitted_clf = clf.fit(X_train, y_train)
    print('max_depth: ',randomforest_fitted_clf.get_params()['max_depth'] )
    print('max_leaf_nodes: ', randomforest_fitted_clf.get_params()['max_leaf_nodes'])
    print('max_features: ', randomforest_fitted_clf.get_params()['max_features'])
    print('n_estimators: ', randomforest_fitted_clf.get_params()['n_estimators'])
    return randomforest_fitted_clf

def score_model(X, Y, model):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.0020, test_size=0.3,
                                                        random_state=27)
    #print('Scoring:  ', pd.Series(y_test).value_counts())
    yhat = model.predict(X_test)
    if(max(yhat)>2):
        y_test = helper.map_to_3class(y_test)
        yhat = helper.map_to_3class(yhat)
    show_results(y_test, yhat)
    f1 = f1_score(y_test, yhat, average='macro')
    return f1

def trim_data(input):
    return input.groupby('clas').filter(lambda x: len(x) > 10000)

def trim_data2(input):
    #return input[input.clas.isin([21.0, 18.0, 7.0, 6.0, 4.0, 5.0, 20.0])]
    return input[np.logical_not(input.clas.isin([8.0]))]

def evaluate_model():
    i = 0
    for scoreConcession in sites:
        print(scoreConcession)
        trainConcessions = list(sites)
        trainConcessions.remove(scoreConcession)
        result = pd.DataFrame(columns = ['concession', 'bands', 'class_scheme', 'score'])
        for key, bands in band_set.items():
            print(key, '....',bands)
            data = pd.DataFrame()
            data_scoring = helper.get_concession_data(bands, scoreConcession)
            data_scoring = trim_data2(data_scoring)
            X_score = data_scoring[[col for col in data_scoring.columns if ((col != 'clas') & (col != 'class_remap'))]]
            X_scaled_score = helper.scale_data(X_score)
            y_score_all = data_scoring['clas'].values

            data = trim_data(helper.get_concession_data(bands, trainConcessions))
            X = data[[col for col in data.columns if ((col != 'clas') & (col != 'class_remap'))]]
            X_scaled = helper.scale_data(X)
            landcover = data['clas'].values
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, landcover, train_size=0.0030, test_size=0.1,
                                                                random_state=16)
            model = train_model(X_train, y_train)
            score_all = score_model(X_scaled_score, y_score_all, model)
            result.loc[i] = [scoreConcession, str(bands), 'ALL', score_all]
            print(result.loc[i])
            i+=1
            # landcover = data['class_remap'].values
            # X_train, X_test, y_train, y_test = train_test_split(X_scaled, landcover, train_size=0.0040, test_size=0.1,
            #                                                     random_state=16)
            model = train_model(X_train, helper.map_to_3class(y_train))
            score_3 = score_model(X_scaled_score, helper.map_to_3class(y_score_all), model)
            result.loc[i] = [scoreConcession, str(bands), '3CLASS', score_3]
            print(result.loc[i])
            i += 1
            scores = [score_all, score_3]
            print('scores:  ',scores)
    print(result)
    result.to_csv(r'/home/eggen/result.csv', index=False)

evaluate_model()
# img=get_feature_inputs(band_set.get(5))
# array=np.asarray(img)
# x = helper.gen_windows(array, pixel_window_size)
# two_class_y, all_class_y = helper.get_landcover_class_image(classConcession)
# y = helper.get_classes(all_class_y, 'clas')
# y2 = helper.get_classes(two_class_y, 'class_binary')
# data = helper.combine_input_landcover(x, y, y2)
# data = data.groupby('clas').filter(lambda x: len(x) > 1000)
# X = data[[col for col in data.columns if ((col != 'clas') & (col != 'class_binary'))]]
# X_scaled = helper.scale_data(X)
