import data_helper as helper
import data.hcs_database as db
import dirfuncs
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier as rfc
import sklearn.metrics
import timer
from sklearn.metrics import f1_score

#############   SETUP  PARAMS    ######################
training_sample_rate = 0.003
resolution = 30
island='Sumatra'
year=str(2015)
sites = [
'app_riau',
'app_oki',
'app_jambi'
    ]
base_dir = dirfuncs.guess_data_dir()
band_set ={
           10:[ helper.bands_base, helper.bands_radar,
                helper.band_evi2 ],
}

pixel_window_size=1
doGridSearch=False


class random_forest_trainer:
    def __init__(self, estimators, depth, max_features, leaf_nodes, bands, scheme):
        self.estimators = estimators
        self.depth = depth
        self.max_features = max_features
        self.leaf_nodes = leaf_nodes
        self.bands = bands
        self.scheme = scheme
        self.model = False

    def train_model(self, X_train, y_train, seed):
        clf = rfc(n_estimators=self.estimators, max_depth=self.depth, max_features=self.max_features, max_leaf_nodes=self.leaf_nodes,
                  # random_state=seed,
                  random_state=13 * seed,
                  oob_score=True, n_jobs=-1,
                  #  class_weight = {0:0.33, 1: 0.33, 2: 0.34})
                  class_weight='balanced')
        if doGridSearch:
            print(" ############  IN GRID SEARCH  ############# ")

            param_grid = [{  # 'max_depth': [14, 16, 18, 20],
                #  'max_leaf_nodes': [14,15,16],
                'max_features': [self.max_features - 0.2, self.max_features - 0.1, self.max_features, self.max_features + .1],
                'n_estimators': [  # 400,
                    self.estimators - 25, self.estimators, self.estimators + 50, self.estimators + 100]
            }]

            grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='f1_macro',
                                       return_train_score=True, refit=True)

            grid_search.fit(X_train, y_train)

            randomforest_fitted_clf = grid_search.best_estimator_
        else:
            randomforest_fitted_clf = clf.fit(X_train, y_train)
        self.model = randomforest_fitted_clf




