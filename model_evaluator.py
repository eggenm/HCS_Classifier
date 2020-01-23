import data_helper as helper
import data.hcs_database as db
import dirfuncs
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier as rfc
import sklearn.metrics
from sklearn.metrics import f1_score

#############   SETUP  PARAMS    ######################
training_sample_rate = 0.003
resolution = 30
sites = [#'gar_pgm',
     'app_jambi',
   'app_riau',
 #  'app_kalbar',
 #        'app_kaltim',

 'app_oki' #,
      # 'crgl_stal'
    ]
base_dir = dirfuncs.guess_data_dir()
band_set ={#1:['bands_radar'],
       #    2: ['bands_base'],
     #       3: ['bands_median'],
      #      4: ['bands_base','bands_radar'],
      #  #    5: ['bands_base','bands_radar','bands_dem']#,
      # #     6: ['bands_radar','bands_dem']
      #      7:['bands_evi2_separate'],
    #        8:['bands_evi2'],
       #     9:['bands_evi2','bands_radar'],
           10:['bands_base','bands_radar','evi2_only'],
      #      11:['bands_base','bands_median','bands_radar','evi2_only']
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
        param_grid = [{'max_depth': [8, 12, 16, 20],
                       'max_leaf_nodes': [5, 10, 15],
                       'max_features': [.25, .5, .75 ],
                       'n_estimators': [100, 250, 500]}]
        grid_search = GridSearchCV(clf, param_grid, cv = 5, scoring = 'f1_macro',
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


def score_model(y_test, yhat):
    show_results(y_test, yhat)
    f1 = f1_score(y_test, yhat, average='macro')
    f1_weighted = f1_score(y_test, yhat, average='weighted')
    return f1,f1_weighted

class model_performance_logger:
    def __init__(self, model, concession, bands, training_concessions):
        """
        Parameters
        ----------
        block: np array
            array drawn from raster using rasterio block read

        fitted_clf: sklearn classifier
            classifier that should be applid to block
        """
        self.model = model
        self.concession = concession
        self.bands = bands
        self.training_concessions = training_concessions
        self.score_type=''
        self.score=np.nan
        self.score_weighted=np.nan
        self.score_2_reclass=np.nan
        self.score_2_reclass_weighted = np.nan


    def set_scores(self, type, score, weighted, two_class, two_class_weighted):
        self.score_type = type
        self.score = score
        self.score_weighted = weighted
        self.score_2_reclass = two_class
        self.score_2_reclass_weighted = two_class_weighted

    def save_score(self):
        print('saved')

def evaluate_model():

    for scoreConcession in sites:
        print(scoreConcession)
        trainConcessions = list(sites)
        trainConcessions.remove(scoreConcession)

        result = pd.DataFrame(columns=['concession', 'bands', 'score_type', 'class_scheme', 'score', 'score_weighted',
                                       'two_class_score', 'two_class_score_weighted', 'training_concessions',
                                       'max_depth',
                                       'max_leaf_nodes', 'max_features', 'n_estimators', 'training_sample_rate', 'resolution'])
        x = range(3, 18, 3)
        for key, bands in band_set.items():
            i = 0
            for y in range(1, 3, 1):
                training_sample_rate = y/1000
                print(key, '....',bands)
                data = pd.DataFrame()
                data_scoring = helper.get_concession_data(bands, scoreConcession)
                data_scoring = helper.trim_data2(data_scoring)
                data_scoring[data_scoring <= -999] = np.nan
                data_scoring = data_scoring.dropna()
                X_score = data_scoring[[col for col in data_scoring.columns if ((col != 'clas') & (col != 'class_remap'))]]
                X_scaled_score = helper.scale_data(X_score)
                y_score_all = data_scoring['clas'].values

                data = helper.trim_data2(helper.get_concession_data(bands, trainConcessions))
                X = data[[col for col in data.columns if ((col != 'clas') & (col != 'class_remap'))]]
                X_scaled = helper.scale_data(X)
                landcover = data['clas'].values
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, landcover, train_size=training_sample_rate, test_size=0.1,
                        random_state=16)

                ##########################################################
                #####     MODEL WITH ALL CLASSES     #########
                model = train_model(X_train, y_train)
                yhat = model.predict(X_scaled_score)
                score_all, score_all_weighted = score_model(helper.map_to_3class(y_score_all), helper.map_to_3class(yhat))
                score_two, score_two_weighted = score_model(helper.map_to_2class(y_score_all), helper.map_to_2class(yhat))
                result.loc[i] = [scoreConcession, str(bands), 'F1', 'ALL', score_all, score_all_weighted, score_two, score_two_weighted, str(trainConcessions),
                                 model.get_params()['max_depth'], model.get_params()['max_leaf_nodes'], model.get_params()['max_features'],model.get_params()['n_estimators'], training_sample_rate, resolution ]
                print(result.loc[i])
                i+=1

                ##########################################################
                #####     MODEL WITH 3 CLASSES     #########
                model = train_model(X_train, helper.map_to_3class(y_train))
                yhat = model.predict(X_scaled_score)
                score_3, score_3_weighted = score_model(helper.map_to_3class(y_score_all), yhat)
                score_two, score_two_weighted = score_model(helper.map_to_2class(y_score_all), helper.map_3_to_2class(yhat))
                result.loc[i] = [scoreConcession, str(bands), 'F1' , '3CLASS', score_3, score_3_weighted, score_two, score_two_weighted, str(trainConcessions),
                                 model.get_params()['max_depth'], model.get_params()['max_leaf_nodes'], model.get_params()['max_features'], model.get_params()['n_estimators'], training_sample_rate, resolution ]
                print(result.loc[i])
                i += 1
            db.save_model_performance(result)
    #print(result)
    #resultfile = base_dir + 'result.csv'
    #result.to_csv(resultfile, index=False)
    print(db.get_all_model_performance())

evaluate_model()
#resultfile = base_dir + 'result.01102020.csv'
#db.get_all_model_performance().to_csv(resultfile, index=False)
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
