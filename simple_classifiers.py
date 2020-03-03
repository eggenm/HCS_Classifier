# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:12:56 2019

@author: rheil
"""
# =============================================================================
# Imports
# =============================================================================
import dirfuncs
import data_helper as helper
import glob
import pandas as pd
from PIL import Image
import numpy as np
import rasterio as rio
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import train_test_split, GridSearchCV
import sklearn.metrics
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import data.hcs_database as db






# =============================================================================
# Identify files
# =============================================================================
base_dir = dirfuncs.guess_data_dir()

concessions = ['app_oki' , 'app_jambi']
island = 'Sumatra'
classConcession = ['app_riau']
year = str(2015)
bands = db.get_best_bands(classConcession)
sample_rate=db.get_best_training_sample_rate(classConcession)

pixel_window_size = 1
iterations = 1
doGridSearch = False
scheme=db.get_best_scheme(classConcession)
estimators = db.get_best_number_estimators(classConcession)
max_features = db.get_best_max_features(classConcession)
depth = db.get_best_max_depth(classConcession)
leaf_nodes = db.get_best_max_leaf_nodes(classConcession)
suffix = 'TEST2___RF_x' + str(iterations) + '_'+scheme + '_'+ str(int(round(sample_rate*10000, 0))) +'_30m_BaseRadarEVI.tif'
#classes = {1: "HCSA",
     #      0: "NA"}

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
# Blockwise predicted map (could be useful for larger maps)
# Probably would need to be modified to work with windowed values rather than multiband image
# =============================================================================
class classify_block:
    def __init__(self, block, randomforest_fitted_clf):
        """
        Parameters
        ----------
        block: np array
            array drawn from raster using rasterio block read

        fitted_clf: sklearn classifier
            classifier that should be applid to block
        """
        self.fitted_clf = randomforest_fitted_clf
        self.shape = block.shape
        block = block.reshape((self.shape[0], self.shape[1] * self.shape[2])).T
        self.block_df = pd.DataFrame(block)
        self.x_df = self.block_df.dropna()

    def classify(self):
        """
        Returns
        -------
        classified: array
            Array of predicted classes
        """
        y_hat = self.fitted_clf.predict(self.x_df)
        y_hat = pd.Series(y_hat, index=self.x_df.index)
        y_hat.name = 'y_hat'
        temp_df = self.block_df.merge(y_hat, left_index=True, right_index=True, how='left')
        classified = temp_df['y_hat'].to_numpy().reshape(self.shape[1], self.shape[2])
        classified = classified[np.newaxis, :, :].astype(rio.int16)
        return classified

    def calc_probabilities(self):
        """
        Returns
        -------
        probabilities: array
            Array of predicted probabilities for each class
        """
        clas_cols = ['prob_' + str(clas) for clas in randomforest_fitted_clf.classes_]
        pred_df = self.fitted_clf.predict_proba(self.x_df)
        pred_df = pd.DataFrame(pred_df, index=self.x_df.index, columns=clas_cols)
        temp_df = self.block_df.merge(pred_df, left_index=True, right_index=True, how='left')
        probabilities = temp_df[clas_cols].to_numpy().T.reshape(len(clas_cols), self.shape[1], self.shape[2])
        probabilities = probabilities.astype(rio.float32)
        return probabilities


def r2(rf, X_train, y_train):
    return r2_score(y_train, rf.predict(X_train))

def forest_importance_ranking(forest):
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()



train_df = helper.trim_data2(helper.get_input_data(bands, island, year, concessions, False))
train_df = helper.drop_no_data(train_df)
X = train_df[[col for col in train_df.columns if  (col != 'clas') ]]

X_scaled = helper.scale_data(X)
landcover = train_df['clas'].values
if(scheme=='3CLASS'):
    landcover=helper.map_to_3class(landcover)
print('VALUE_COUNTS:  ',pd.Series(landcover).value_counts(dropna=False))
predictableClasses = pd.Series(landcover).value_counts(dropna=False).index.tolist()
print(predictableClasses)
#encoder = LabelEncoder() # Could be used if we're not always dealing with the same classes
#encoder.fit(y)
#n_classes = encoder.classes_.shape[0]
#y_encoded = encoder.transform(y)

predictions = pd.DataFrame()
probabilities_dict = {}


df_class = helper.get_input_data(bands, island, year, classConcession, True)
#df_class=helper.trim_data2(df_class)
X_class = df_class[[col for col in df_class.columns if  (col != 'clas') ]]
X_scaled_class = helper.scale_data(X_class)
X_class=0
y_class = df_class['clas'].values
if(scheme=='3CLASS'):
    y_class=helper.map_to_3class(y_class)

for key in predictableClasses:
    probabilities_dict[key]=pd.DataFrame()
for seed in range(1,iterations+1):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, landcover, train_size=sample_rate, test_size=0.1,
                                                        random_state=13*seed)


    # # =============================================================================
    # # Train and test random forest classifier
    # # =============================================================================



    clf = rfc(n_estimators=estimators, max_depth = depth, max_features = max_features, max_leaf_nodes = leaf_nodes,
              #random_state=seed,
              random_state=13*seed,
              oob_score = True, n_jobs = -1,
            #  class_weight = {0:0.33, 1: 0.33, 2: 0.34})
              class_weight ='balanced')
    if doGridSearch:
        print(" ############  IN GRID SEARCH  ############# ")

        param_grid = [{#'max_depth': [14, 16, 18, 20],
                     #  'max_leaf_nodes': [14,15,16],
                       'max_features': [max_features-0.2, max_features-0.1, max_features, max_features+.1],
                       'n_estimators': [ #400,
                                        estimators-25, estimators, estimators+50, estimators+100]
        }]

        grid_search = GridSearchCV(clf, param_grid, cv=5,  scoring = 'f1_macro',
                                   return_train_score=True, refit=True)

        grid_search.fit(X_train, y_train)

        randomforest_fitted_clf = grid_search.best_estimator_
    else:
        randomforest_fitted_clf = clf.fit(X_train, y_train)
    y_hat = randomforest_fitted_clf.predict(X_test)

    print('max_depth: ', randomforest_fitted_clf.get_params()['max_depth'])
    print('max_leaf_nodes: ', randomforest_fitted_clf.get_params()['max_leaf_nodes'])
    print('max_features: ', randomforest_fitted_clf.get_params()['max_features'])
    print('n_estimators: ', randomforest_fitted_clf.get_params()['n_estimators'])
    print('*************  RANDOM FOREST  - X_TEST  **********************')
    report = sklearn.metrics.classification_report(y_test, y_hat)
    print(report)
    #export_report = pd.DataFrame(report).to_csv(r'C:\Users\ME\Desktop\export_report_riau.csv', index=None,
        #                           header=True)
    confMatrix= sklearn.metrics.confusion_matrix(y_test, y_hat)
    print(confMatrix)
   # export_csv = pd.DataFrame(confMatrix).to_csv(r'C:\Users\ME\Desktop\export_confusion_riau.csv', index=None,
      #                     header=True)  # Don't forget to add '.csv' at the end of the path

    y_hat = randomforest_fitted_clf.predict(X_train)
    print('*************  RANDOM FOREST  - X_TRAIN  **********************')
    print(sklearn.metrics.classification_report(y_train, y_hat))
    print(sklearn.metrics.confusion_matrix(y_train, y_hat))

    #perm_imp_rfpimp = permutation_importances(randomforest_fitted_clf, X_train, y_train, r2)
    #print(perm_imp_rfpimp)
   # forest_importance_ranking(randomforest_fitted_clf)
    # =============================================================================
    # Neural network
    # =============================================================================
    # nueral_model = keras.models.Sequential()
    # nueral_model.add(keras.layers.InputLayer(input_shape = (windows.shape[1],)))
    # nueral_model.add(keras.layers.Dense(30, activation="relu"))
    # nueral_model.add(keras.layers.Dense(30, activation="relu"))
    # nueral_model.add(keras.layers.Dense(30, activation="relu"))
    # nueral_model.add(keras.layers.Dense(30, activation="relu"))
    # nueral_model.add(keras.layers.Dense(3, activation="softmax"))
    #
    # nueral_model.compile(optimizer=keras.optimizers.Adadelta(),
    #             loss='sparse_categorical_crossentropy',
    #           #  metrics=['sparse_categorical_accuracy'])
    #            #   optimizer=keras.optimizers.SGD(lr=0.05),
    #               metrics=["accuracy"])
    # history = nueral_model.fit(X_train, y_train, epochs=1,
    #                     validation_split = 0.5,
    #                     callbacks=[keras.callbacks.EarlyStopping(patience=5)])
    #
    # nueral_model.evaluate(X_test, y_test)
    # y_proba = nueral_model.predict(X_test)
    # y_predict = nueral_model.predict_classes(X_test)
    # print('*************  NEURAL NET REPORT *********************')
    # print(sklearn.metrics.classification_report(y_test, y_predict))
    # print(sklearn.metrics.confusion_matrix(y_test, y_predict))


    # =============================================================================
    # Neural net with hyperparameter tuning
    # =============================================================================
    # def build_model(n_hidden = 1, n_neurons = 30, learning_rate = 0.03, n_bands = 13, n_classes = 3):
    #     model = keras.models.Sequential()
    #     model.add(keras.layers.InputLayer(input_shape = (n_bands,)))
    #     for layer in range(n_hidden):
    #         model.add(keras.layers.Dense(n_neurons, activation="relu"))
    #     model.add(keras.layers.Dense(n_classes, activation="softmax"))
    #     model.compile(loss="sparse_categorical_crossentropy",
    #                   optimizer=keras.optimizers.SGD(lr=learning_rate),
    #                   metrics=["accuracy"])
    #     return model
    #
    # keras_clf = keras.wrappers.scikit_learn.KerasClassifier(build_model)
    #
    # param_distribs = {"n_hidden": [1, 5, 10],
    #                   "n_neurons": np.arange(1, 100),
    #                   "learning_rate": reciprocal(3e-3, 7e-2)}
    # rnd_search_cv = RandomizedSearchCV(keras_clf, param_distribs, n_iter=10, cv=3)
    # rnd_search_cv.fit(X_train, y_train, epochs=100, validation_split = 0.8,
    #                   callbacks = [keras.callbacks.EarlyStopping(patience = 5)])
    #
    # keras_clf.fit(X_train, y_train, epochs=10, validation_split = 0.8,
    #               callbacks = [keras.callbacks.EarlyStopping(patience = 5)])
    #
    # nueral_model = rnd_search_cv.best_estimator_
    # y_pred = nueral_model.predict(X_train)
    # print('*************  TUNED NEURAL NET - WITHIN REPORT *********************')
    # print(sklearn.metrics.classification_report(y_train, y_pred))
    # print(sklearn.metrics.confusion_matrix(y_train, y_pred))
    #
    # y_pred = nueral_model.predict(X_test)
    # print('*************  TUNED NEURAL NET - OUT REPORT *********************')
    # print(sklearn.metrics.classification_report(y_test, y_pred))
    # print(sklearn.metrics.confusion_matrix(y_test, y_pred))

    # =============================================================================
    # Create predicted map
    # =============================================================================

    X=0
    y_test=0
    X_train=0
    y_train=0
    y_hat=0

    X_test=0
    if (scheme == 'ALL'):
        predictions[seed] = helper.map_to_2class(randomforest_fitted_clf.predict(X_scaled_class))
    else:
        predictions[seed] = helper.map_3_to_2class(randomforest_fitted_clf.predict(X_scaled_class))

    #tempProb=randomforest_fitted_clf.predict_proba(X_scaled_class)[:,:]
    print(randomforest_fitted_clf.classes_)
    #print(tempProb.shape)
    #print(tempProb[:,16])
    #tempDf = pd.DataFrame(tempProb)
    #for i, key in enumerate(randomforest_fitted_clf.classes_):
       # probabilities_dict[key][seed] = tempProb[:,i]

X_scaled=0
X_scaled_class=0
if iterations>1:
    temp = pd.DataFrame(predictions.mode(axis=1)[0])
else:
    temp=predictions

outclas_file = base_dir + classConcession[0] + '/sklearn_test/classified' + suffix
referencefile = base_dir + classConcession[0] + '/' + classConcession[0] + '_all_class.remapped.tif'
prob_file = base_dir + classConcession[0] +  '/sklearn_test/prob_file' + suffix
print(referencefile)
file_list = sorted(glob.glob(referencefile))
with rio.open(file_list[0]) as src:
    height = src.height
    width = src.width
    shape = src.shape
    crs = src.crs
    transform = src.transform
    dtype = rio.int16
    count = 1
    full_index = pd.MultiIndex.from_product([range(shape[0]), range(shape[1])], names=['i', 'j'])
    temp = temp.set_index(full_index)
    print(shape[0])
    print(shape[1])
    print(full_index.size)
    #for key in randomforest_fitted_clf.classes_:
        # if iterations > 1:
        #     tempMean = probabilities_dict[key].mean(axis=1)
        # else:
        #     tempMean = probabilities_dict[key]
        #probabilities_dict[key] = pd.DataFrame(tempMean).set_index(full_index)
        #probabilities_dict[key] = probabilities_dict[key].values.reshape(shape[0], shape[1])



    df_class['predicted'] = temp
    clas_df = pd.DataFrame(index = full_index)
    classified = clas_df.merge(df_class['predicted'], left_index=True, right_index=True, how='left').sort_index()

    classified = classified.values.reshape(shape[0], shape[1])
    # clas_img = ((classified * 255) / 2).astype('uint8')
    # clas_img = helper.mask_water(clas_img, classConcession)
    # clas_img = Image.fromarray(clas_img)
    # clas_img.show()

    print('*************  RANDOM FOREST  - ACTUAL  **********************')

    df_class[df_class <= -999] = np.nan
    df_class = helper.drop_no_data(df_class)
    print('ACTUAL:  ', df_class['clas'].value_counts())
    df_class['clas'] = helper.map_to_2class(df_class['clas'])
    print('ACTUAL:  ', df_class['clas'].value_counts())
    print('***scheme:  ' ,scheme)

    print('Predicted:  ', df_class['predicted'].value_counts())
    #print('ACTUAL:  ', df_class['clas'].value_counts())

    print(sklearn.metrics.classification_report(df_class['clas'], df_class['predicted']))
    print(sklearn.metrics.confusion_matrix(df_class['clas'], df_class['predicted']))
    classified = classified[np.newaxis, :, :].astype(rio.int16)

    # score_all, score_all_weighted = helper.score_model(helper.map_to_3class(df_class['clas']), helper.map_to_3class(df_class['predicted']))
    # score_two, score_two_weighted = helper.score_model(helper.map_to_2class(df_class['clas']), helper.map_to_2class(df_class['predicted']))
    # result.loc[i] = [scoreConcession, str(bands), 'F1', 'ALL', score_all, score_all_weighted, score_two,
    #                  score_two_weighted, str(trainConcessions),
    #                  model.get_params()['max_depth'], model.get_params()['max_leaf_nodes'],
    #                  model.get_params()['max_features'], model.get_params()['n_estimators'], training_sample_rate,
    #                  resolution]
    # print(result.loc[i])
    band=0

    clas_dst = rio.open(outclas_file, 'w', driver = 'GTiff',
                      height = height, width = width,
                      crs = crs, dtype = dtype,
                      count = count, transform = transform)
    #prob_dst = rio.open(prob_file, 'w', driver = 'GTiff',
                       # height = src.height, width = src.width,
                       # crs = src.crs, dtype = rio.float32,
                       # count = len(randomforest_fitted_clf.classes_), transform = src.transform)
    for ji, window in src.block_windows(1):
        print('ji:  ', ji)
        print('window:  ', window)
        block = src.read(window = window)
        if sum(sum(sum(~np.isnan(block))))>0:
                block_classifier = classify_block(block, randomforest_fitted_clf)
                classified = block_classifier.classify()
                probabilities = block_classifier.calc_probabilities()
                clas_dst.write(classified, window = window)
         #       prob_dst.write(probabilities, window = window)

 # with rio.open(outclas_file, 'w', driver = 'GTiff',
    #               height = height, width = width,
    #               crs = crs, dtype = dtype,
    #               count = count, transform = transform) as clas_dst:
    #     clas_dst.write(classified)
    # with rio.open(prob_file, 'w', driver = 'GTiff',
    #               height = height, width = width,
    #               crs = crs, dtype = rio.float32,
    #               count = len(randomforest_fitted_clf.classes_), transform = transform) as prob_dst:
    #     for key, value in probabilities_dict.items():
    #         print(key, '....')
    #         prob_dst.write_band(band+1, value.astype(rio.float32))
#prob_dst.close()
clas_dst.close()
#prob_dst.close()
src.close()

#clas_file = base_dir + classConcession +  '/sklearn_test/class_file.tif'
prob_file = base_dir + classConcession[0] +  '/sklearn_test/prob_file' + suffix

# with rio.open(outtif) as src:
#     clas_dst = rio.open(outclas_file, 'w', driver = 'GTiff',
#                   height = height, width = width,
#                   crs = crs, dtype = dtype,
#                   count = count, transform = transform)
#     prob_dst = rio.open(prob_file, 'w', driver = 'GTiff',
#                    height = src.height, width = src.width,
#                    crs = src.crs, dtype = rio.float32,
#                    count = len(randomforest_fitted_clf.classes_), transform = src.transform)
#     for ji, window in src.block_windows(1):
#         block = src.read(window = window)
#         if sum(sum(sum(~np.isnan(block))))>0:
#             block_classifier = classify_block(block, randomforest_fitted_clf)
#             classified = block_classifier.classify()
#             probabilities = block_classifier.calc_probabilities()
#             clas_dst.write(classified, window = window)
#             prob_dst.write(probabilities, window = window)
#
# clas_dst.close()
# prob_dst.close()
