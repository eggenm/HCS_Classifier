from builtins import dict

import data_helper as helper
import data.hcs_database as db
import dirfuncs
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier as rfc
import sklearn.metrics
from sklearn.metrics import f1_score
from copy import deepcopy
import timer
import itertools
import sampler

#############   SETUP  PARAMS    ######################
training_sample_rate = 0.003
resolution = 30

year=str(2015)
sites = [ #'app_muba':'Sumatra',
#['app_riau'],
#['app_oki'],
  #   ['app_jambi'] ,
# ['Bumitama_PTDamaiAgroSejahtera'],
    ['Bumitama_PTHungarindoPersada'],
    ['PTAgroAndalan'],
    ['PTMitraNusaSarana'],
    ['gar_pgm'],
 ['Bumitama_PTGemilangMakmurSubur'],
 #         'crgl_stal' : 'Sumatra',

#'app_kalbar':'Kalimantan','app_kaltim':'Kalimantan',
   #       'Bumitama_PTDamaiAgroSejahtera':'Kalimantan',
   #
   #  'PTAgroAndalan':'Kalimantan',
   #    'PTMitraNusaSarana':'Kalimantan',
   # 'gar_pgm':'Kalimantan',
#'Bumitama_PTGemilangMakmurSubur':'Kalimantan' ,
    # 'Bumitama_PTHungarindoPersada':'Kalimantan',
#['Bumitama_PTDamaiAgroSejahtera','PTAgroAndalan'],
#['PTMitraNusaSarana', 'gar_pgm'],
#['Bumitama_PTGemilangMakmurSubur', 'Bumitama_PTHungarindoPersada']

          ]
#sites = [

#    ]
base_dir = dirfuncs.guess_data_dir()
band_set ={ # 0: ['blue_max', 'green_max', 'red_max', 'nir_max', 'swir1_max', 'swir2_max', 'EVI' ],
#             1: ['blue_max', 'green_max', 'red_max', 'nir_max', 'swir1_max', 'swir2_max', 'EVI', 'slope'],
#             2:['blue_max', 'green_max', 'red_max', 'nir_max', 'swir1_max', 'swir2_max', 'VH', 'VV' ,'EVI', 'slope'] , #'elevation', , 'aspect']
#             3:['blue_max', 'green_max', 'red_max', 'nir_max', 'swir1_max', 'swir2_max', 'VH_0', 'VV_0', 'EVI' , 'slope'],
#             4:['blue_max', 'green_max', 'red_max', 'nir_max', 'swir1_max', 'swir2_max', 'VH_2', 'VV_2', 'EVI', 'slope'],
#           #  5:['blue_max', 'green_max', 'red_max', 'nir_max', 'swir1_max', 'swir2_max', 'VH', 'VV', 'VH_0', 'VV_0', 'EVI' ],
#           #  6:['blue_max', 'green_max', 'red_max', 'nir_max', 'swir1_max', 'swir2_max', 'VH', 'VV', 'VH_2', 'VV_2', 'EVI' ],
#          #   7:['blue_max', 'green_max', 'red_max', 'nir_max', 'swir1_max', 'swir2_max', 'VH', 'VV','VH_0', 'VV_0', 'VH_2', 'VV_2', 'EVI' ],
#             8: ['blue_max', 'green_max', 'red_max', 'nir_max', 'swir1_max', 'swir2_max', 'VH', 'VV', 'VH_0', 'VV_0',
#                 'VH_2', 'VV_2', 'EVI', 'slope'],
            9: [ 'swir1_max',  'VH', 'VV', 'VV_2', 'VH_2', 'EVI' ,'slope'],
            91: [ 'swir1_max',  'VH', 'VV', 'VV_2', 'VH_2', 'EVI' ],
            94: ['nir_max', 'swir1_max',   'VH', 'VV', 'VV_2', 'VH_2','slope'],
            95: ['nir_max', 'swir1_max',   'VH', 'VV', 'VV_2', 'VH_2'],
            96: ['nir_max', 'swir1_max',   'VV_2', 'VH_2', 'EVI' ,'slope'],
            97: ['nir_max', 'swir1_max',   'VV_2', 'VH_2', 'EVI' ],
            98: ['nir_max', 'swir1_max',  'VV', 'VH', 'EVI' ,'slope'],
            99: ['nir_max', 'swir1_max',  'VV', 'VH', 'EVI'],
            19: ['swir1_max',  'VH', 'VV', 'VV_2', 'VH_2', 'EVI', 'slope'],
            29: ['swir1_max', 'VH', 'VV', 'VV_2', 'VH_2', 'EVI' ],
            39: ['swir1_max',  'VH', 'VV', 'EVI', 'slope'],
            49: ['swir1_max',  'VH', 'VV', 'EVI', 'nir_max'],
            10: ['swir1_max', 'EVI', 'VH' , 'VV_2'],
            11: ['swir1_max', 'slope', 'VH', 'VV_2'],
            12: ['swir1_max',  'VH_2', 'VH', 'nir_max'],
            13: ['swir1_max', 'EVI', 'VH', 'VV'],
            14: ['swir1_max', 'slope', 'VH', 'VV', 'VH_2'],
            16: ['nir_max', 'swir1_max',   'VH', 'VV', 'VV_2', 'VH_2','EVI','slope'],
            17: ['nir_max', 'swir1_max',   'VH', 'VV', 'VV_2', 'VH_2', 'EVI',],
            18: ['nir_max', 'swir1_max', 'swir2_max', 'VH_0', 'VV_0', 'EVI'],
            19: ['nir_max', 'swir1_max', 'swir2_max', 'VH', 'VV', 'EVI'],
            20: ['nir_max', 'swir1_max', 'swir2_max', 'VH', 'VV', 'EVI', 'slope']
             #    'EVI', 'slope']
            }

add_1_band_set = {
    'base': ['swir1_max', 'VV_2', 'slope'],
    'nir': ['swir1_max', 'VV_2', 'slope','nir_max'],
    'swir2': ['swir1_max', 'VV_2', 'slope','swir2_max'],
    'VH': ['swir1_max', 'VV_2', 'slope','VH'],
    'VH2': ['swir1_max', 'VV_2', 'slope','VH_2'],
    'VV0': ['swir1_max', 'VV_2', 'slope','VV_0'],
    'VV': ['swir1_max', 'VV_2', 'slope','VV'],
    'VH0': ['swir1_max', 'VV_2', 'slope','VH_0'],
    'EVI': ['swir1_max', 'VV_2', 'slope','EVI'],
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
    kappa  = sklearn.metrics.cohen_kappa_score(y_test, y_hat)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_hat)
    f1 = f1_score(y_test, y_hat, average='macro')
    f1_weighted = f1_score(y_test, y_hat, average='weighted')
    print('KAPPA:  ', kappa)
    print('Accuracy:  ', accuracy)
    print('f1:  ', f1)
    print('f1_weighted:  ', f1_weighted)


def train_model(X_train, y_train, score_stat):
    #print('Training:  ', pd.Series(y_train).value_counts())
    #print(X_train[:7])
    clf = rfc(n_estimators=40, max_depth=8, max_features=.95, max_leaf_nodes=18,
              random_state=16, oob_score=True, n_jobs=-1,
              #  class_weight = {0:0.33, 1: 0.33, 2: 0.34})
              class_weight='balanced')
    if doGridSearch:
        print(" ############  IN GRID SEARCH  ############# ")
        param_grid = [{#'max_depth': [ 6, 8, 10 ],
                       'max_leaf_nodes': [6, 10 ],
                       'max_features': [ #.2,
                                         .33, .65,
                                        .8 ],
                       'n_estimators': [100, 250, 375, 500, 750]}]

     #   param_grid = [{
      #                   'max_leaf_nodes': [6, 10],
       #                 'n_estimators': [100, 250, 375, 500 ]}]
        grid_search = GridSearchCV(clf, param_grid, cv = 5, scoring = score_stat,
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
    kappa = sklearn.metrics.cohen_kappa_score(y_test, yhat)
    return f1,f1_weighted, kappa

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

def init_x_y_data(sites, band_set):
    try:
        with timer.Timer() as t:
            for concessions in sites:
                for key, bands in band_set.items():
                    new_key = str(key) + str(concessions[0])  # assumes unique concession 0  !!!
                    print('NEW_KEY:  ', new_key)
                    try:
                        raw_class_data[concessions[0]]
                        #if we already have class data for the concession just return the predictors
                        print("****WE HAVE THE RAW CONCESSION CLASSES ****")
                        data_scoring = helper.get_input_data(bands, year, concessions, True)
                        data_scoring['clas'] = raw_class_data[concessions[0]]
                    except KeyError:
                        data_scoring = helper.get_input_data(bands, year, concessions, False)
                        raw_class_data[concessions[0]] = data_scoring['clas'].values
                    data_scoring = helper.trim_data2(data_scoring)
                    data_scoring = helper.drop_no_data(data_scoring)
                    X_score = data_scoring[[col for col in data_scoring.columns if ((col != 'clas') & (col != 'class_remap'))]]
                    y_score_all = data_scoring['clas'].values
                    actual_data[new_key] = deepcopy(y_score_all)
                    scaled_x_data[new_key] = deepcopy(X_score)
    finally:
        print('INIT_XY_DATA took %.03f sec.' % t.interval)
        X_scaled_score = False
        X_score = False
        y_score_all = False

def evaluate_model():

    for scoreConcession in sites:
        print(scoreConcession)
        trainConcessions = deepcopy(sites)
        trainConcessions.remove(scoreConcession)
        trainConcessions = [item for sublist in trainConcessions for item in sublist]

        result = pd.DataFrame(columns=['concession', 'bands', 'score_type', 'class_scheme', 'score', 'score_weighted',
                                       'two_class_score', 'two_class_score_weighted', 'training_concessions',
                                       'max_depth',
                                       'max_leaf_nodes', 'max_features', 'n_estimators', 'training_sample_rate', 'resolution', 'kappa', 'kappa_3'])
        x = range(4, 11, 2)
        for key, bands in band_set.items():
            myBands = {key:bands}
            init_x_y_data(sites, myBands)

            i = 0

            print(key, '....',bands)

            X_scaled_score = get_predictor_data(key,scoreConcession)

            y_score_all = get_landcover_data(key,scoreConcession)
            X_scaled = get_predictor_data2(key, trainConcessions)
            landcover = get_landcover_data2(key,trainConcessions)
            for y in range(400, 700, 125):
                sample_sizes_dict = sampler.Sampler.get_sample_rate_by_type(y, sites)
                for concession in trainConcessions:
                    train_sample = sample_sizes_dict[db.data_context_dict[concession][0]]
                    test_sample = sample_sizes_dict[db.data_context_dict[concession][1]]
                    X_train_site, X_test_site, y_train_site, y_test_site = train_test_split(X_scaled[concession], landcover[concession], train_size=train_sample, test_size=test_sample,
                            random_state=16)
                    print('****  training_sample_rate  *****', train_sample)
                    print('****  X_train size *****', len(X_train))
                ##########################################################
                #####     MODEL WITH ALL CLASSES     #########
                # model = train_model(X_train, y_train.values.ravel())
                # yhat = model.predict(X_scaled_score)
                # print('PREDICTED:  ', pd.Series(yhat).value_counts())
                # score_all, score_all_weighted = score_model(helper.map_to_3class(y_score_all.values.ravel()), helper.map_to_3class(yhat))
                # score_two, score_two_weighted = score_model(helper.map_to_2class(y_score_all.values.ravel()), helper.map_to_2class(yhat))
                # result.loc[i] = [scoreConcession, str(bands), 'F1', 'ALL', score_all, score_all_weighted, score_two, score_two_weighted, str(trainConcessions),
                #                  model.get_params()['max_depth'], model.get_params()['max_leaf_nodes'], model.get_params()['max_features'],model.get_params()['n_estimators'], training_sample_rate, resolution ]
                # print(result.loc[i])
                # i+=1

                ##########################################################
                #####     MODEL WITH 3 CLASSES     #########

                ###########   USE ROC_AUC  ###################

                # model = train_model(X_train, helper.map_to_3class(y_train.values.ravel()), 'roc_auc_ovo')
                # yhat = model.predict(X_scaled_score)
                # score_3, score_3_weighted, kappa3 = score_model(helper.map_to_3class(y_score_all.values.ravel()), yhat)
                # score_two, score_two_weighted, kappa2 = score_model(helper.map_to_2class(y_score_all.values.ravel()),
                #                                             helper.map_3_to_2class(yhat))
                # result.loc[i] = [scoreConcession[0], str(bands), 'roc_auc_ovo', '3CLASS', score_3, score_3_weighted, score_two,
                #                  score_two_weighted, str(trainConcessions),
                #                  model.get_params()['max_depth'], model.get_params()['max_leaf_nodes'],
                #                  model.get_params()['max_features'], model.get_params()['n_estimators'],
                #                  training_sample_rate, resolution, kappa2, kappa3]
                # print(result.loc[i])
                # i += 1

                ###########   USE F1  ###################

                model = train_model(X_train, helper.map_to_3class(y_train.values.ravel()), 'f1_macro')
                yhat = model.predict(X_scaled_score)
                score_3, score_3_weighted, kappa3 = score_model(helper.map_to_3class(y_score_all.values.ravel()), yhat)
                score_two, score_two_weighted, kappa2 = score_model(helper.map_to_2class(y_score_all.values.ravel()), helper.map_3_to_2class(yhat))
                result.loc[i] = [scoreConcession[0], str(bands), 'F1' , '3CLASS', score_3, score_3_weighted, score_two, score_two_weighted, str(trainConcessions),
                                 model.get_params()['max_depth'], model.get_params()['max_leaf_nodes'], model.get_params()['max_features'], model.get_params()['n_estimators'], training_sample_rate, resolution, kappa2, kappa3]
                print(result.loc[i])
                i += 1
            db.save_model_performance(result)
    #print(result)trim_data2
    #resultfile = base_dir + 'result.csv'
    #result.to_csv(resultfile, index=False)
    print(db.get_all_model_performance())

def evaluate_bands():
    i = 0
    result = pd.DataFrame(
        columns=['concession', 'added_band', 'two_class_score_weighted_addl', 'kappa_addl', 'kappa_3_addl'])
    for concession in sites:
        trainConcessions = concession


        score_base = 0.0
        kappa2base = 0.0
        kappa3base = 0.0
        for name, bands in add_1_band_set.items():
            myBands = {name: bands}
            init_x_y_data([concession], myBands)
            X_scaled = get_predictor_data(name, trainConcessions)
            landcover = get_landcover_data(name, trainConcessions)
            training_sample_rate = 350
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, landcover, train_size=training_sample_rate,
                                                                test_size=0.30,
                                                                random_state=33)
            model = train_model(X_train, helper.map_to_3class(y_train.values.ravel()), 'f1_macro')
            yhat = model.predict(X_test)
            if(name =='base'):
                score_3_na, score_3_weighted_na, kappa3base = score_model(helper.map_to_3class(y_test.values.ravel()), yhat)
                score_two_na, score_base, kappa2base = score_model(helper.map_to_2class(y_test.values.ravel()),
                                                                    helper.map_3_to_2class(yhat))
                print(concession, ' score_base:  ', score_base )
                print(concession, ' kappa3base:  ', kappa3base)
                print(concession, ' kappa2base:  ', kappa2base)
                continue
            elif(score_base==0):
                raise RuntimeError

            score_3, score_3_weighted, kappa3 = score_model(helper.map_to_3class(y_test.values.ravel()), yhat)
            score_two, score_two_weighted, kappa2 = score_model(helper.map_to_2class(y_test.values.ravel()),
                                                                helper.map_3_to_2class(yhat))
            print(concession,' ', name, ' score:  ', score_two_weighted)
            print(concession,' ', name, ' kappa3:  ', kappa3)
            print(concession, ' ', name,' kappa2:  ', kappa3)
            result.loc[i] = [concession, name, score_two_weighted-score_base, kappa2-kappa2base, kappa3-kappa3base]
            print(result.loc[i])
            i += 1
    return result


def get_predictor_data(band_id, concessions):
        new_key = str(band_id)+concessions[0]
        data = pd.DataFrame(scaled_x_data[new_key])
        return data

def get_landcover_data(band_id, concessions):
    data = pd.DataFrame()
    new_key = str(band_id) + concessions[0]
    data = pd.DataFrame(actual_data[new_key])
    return data

def get_predictor_data2(band_id, concessions):
    data = pd.DataFrame()
    for concession in concessions:
        new_key = str(band_id)+concession
        if data.empty:
            data = pd.DataFrame(scaled_x_data[new_key])
        else:
            data = pd.concat([data, pd.DataFrame(scaled_x_data[new_key])], ignore_index=True)
    return data

def get_landcover_data2(band_id, concessions):
    data = pd.DataFrame()
    for concession in concessions:
        new_key = str(band_id) + concession
        if data.empty:
            data = pd.DataFrame(actual_data[new_key])
        else:
            data = pd.concat([data, pd.DataFrame(actual_data[new_key])], ignore_index=True)
    return data

if __name__ == "__main__":
    scaled_x_data = dict()
    actual_data = dict()
    raw_class_data = dict()
    #init_x_y_data(sites, band_set)
    resultfile = base_dir + 'Suma_result.06012020.csv'
    evaluate_model()

    #resultfile = base_dir + 'add1band_result.05292020.csv'
    db.get_all_model_performance().to_csv(resultfile, index=False)
    #evaluate_bands().to_csv(resultfile, index=False)
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
