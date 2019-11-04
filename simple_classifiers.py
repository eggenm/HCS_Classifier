# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:12:56 2019

@author: rheil
"""
# =============================================================================
# Imports
# =============================================================================
import ee
ee.Initialize()
import keras
import dirfuncs
import hcs_database as hcs_db
import glob
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
from osgeo import gdal,gdalconst
import rasterio as rio
import rasterio.warp
from rasterio.mask import mask
import rasterio.crs
from rasterio.warp import calculate_default_transform, reproject, Resampling
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import train_test_split, cross_val_score, \
    GridSearchCV, cross_val_predict, ShuffleSplit, learning_curve, RandomizedSearchCV
import sklearn.metrics
from scipy.stats import reciprocal
from sklearn.preprocessing import StandardScaler, LabelEncoder
import keras.wrappers.scikit_learn
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
from rfpimp import permutation_importances
import matplotlib.pyplot as plt






# =============================================================================
# Identify files
# =============================================================================
base_dir = dirfuncs.guess_data_dir()
concessions = [ 'app_oki']
classConcession = 'app_riau'
pixel_window_size = 1
iterations = 1
suffix = 'RF_x' + str(iterations) + '_at_200_44BandInput_plus_FeatImp.tif'
stackData = True
doGridSearch = True

classes = {1: "HCSA",
           0: "NA"}

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
    shift = (n-1)/2
    window = img[:, int(i-shift):int(i+shift+1), int(j-shift):int(j+shift+1)]
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
    start = int((n-1)/2)
    end_i = shape[1] - start
    end_j = shape[2] - start
    win_dict = {}
    for i in range(start, end_i):
        for j in range(start, end_j):
            win_dict[(i,j)] = return_window(array, i, j, n)
    windows = pd.Series(win_dict)
    windows.index.names = ['i', 'j']
    index = windows.index
    windows = pd.DataFrame(windows.apply(lambda x: x.flatten()).values.tolist(), index = index)
    return(windows)


def stack_image_input_data(concession):
    input_dir = base_dir + concession + "/in/"
    print(input_dir)
    outtif = base_dir + concession + '/out/input_' + concession + '.tif'
    if stackData:
        file_list = sorted(glob.glob(input_dir+"/*.tif"))
        with rasterio.open(file_list[0]) as src0:
            meta = src0.meta

        # Update meta to reflect the number of layers
        meta.update(count=len(file_list), dtype='float64')

        # Read each layer and write it to stack
        with rasterio.open(outtif, 'w', **meta) as dst:
            for i, layer in enumerate(file_list, start=1):
                print(i, '....', layer)
                with rasterio.open(layer) as src1:
                    band = src1.read(1).astype('float64')
                   # print('Max:  ', band.max())
                   # print('Min:  ', band.min())
                    dst.write_band(i, band)
        dst.close()
    return outtif

def get_landcover_class_image(concession):
    clas_file = base_dir + concession + '/' + concession + '_remap_2class.remapped.tif'
    print(clas_file)
    file_list = sorted(glob.glob(clas_file))
    ## Read classification labels
    with rio.open(file_list[0]) as clas_src:
        clas = clas_src.read()
    return clas

def get_classes(classImage):
    clas_dict = {}
    shape=classImage.shape
    for i in range(classImage.shape[1]):
        for j in range(classImage.shape[2]):
            clas_dict[(i, j)] = classImage[0, i, j]
    full_index = pd.MultiIndex.from_product([range(shape[1]), range(shape[2])], names=['i', 'j'])
    classes = pd.DataFrame({'class': pd.Series(clas_dict)}, index=full_index)
    return classes

def combine_input_landcover(input, landcover):
    data_df = landcover.merge(input, left_index=True, right_index=True, how='left')
    data_df[data_df <= -999] = np.nan
    data_df = data_df.dropna()
    print('*****data_df shape:  ', data_df.shape)
    return data_df

def scale_data(x):
    print('x min:  ', x.min())
    print('xmax:  ', x.max())
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x.astype(np.float64))
    print('x_scaled min:  ',x_scaled.min())
    print('x_scaled max:  ', x_scaled.max())
    return x_scaled

def mask_water(an_img, concession):
    with rio.open(base_dir + concession + "/in/" + concession + "_radar.VH_2015.tif") as radar1:
        radar = radar1.read()
    watermask = np.empty(radar.shape, dtype=rasterio.uint8)
    watermask = np.where(radar > -17.85, 1, -999999).reshape(radar.shape[1], radar.shape[2])
    water_img = Image.fromarray(255 * watermask.astype('uint8'))
    water_img.show()
    print(an_img.max())
    print(an_img.min())
    an_img = an_img*watermask
    print(an_img.max())
    print(an_img.min())
    return an_img

def get_all_concession_data(concessions):
    allInput = np.empty((0), 'float64')
    landcover = np.empty((0), 'int')
    data = pd.DataFrame()
    for concession in concessions:
        outtif = stack_image_input_data(concession)
        with rio.open(outtif) as img_src:
            img = img_src.read()
            x = gen_windows(img, pixel_window_size)
            print('x.shape:  ', x.shape)
          #  allInput = np.append(allInput, scale_data(x))
          #  print('allInput.shape:  ',allInput.shape)
        class_image = get_landcover_class_image(concession)
        class_image = mask_water(class_image, concession)
        y = get_classes(class_image)
        print('y.shape:  ', y.shape)
        if data.empty:
            data=combine_input_landcover(x,y)
        else:
            data = pd.concat([data, combine_input_landcover(x,y)], ignore_index=True)
            print("  data.shape:  ", data.shape)
    return data

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

def extractData(X_scaled, y, imagepath, concession):

    with rio.open(imagepath) as img_src:
        img = img_src.read()

    shape = img.shape
    print("shape:  ", shape)
    windows = gen_windows(img, 1)
    clas_file = base_dir + concession + '/' + concession + '_remap_2class.remapped.tif'
    print(clas_file)
    file_list = sorted(glob.glob(clas_file))
    ## Read classification labels
    with rio.open(file_list[0]) as clas_src:
        clas = clas_src.read()
    # with rio.open(base_dir + concession + "/in/" + concession +"radar.VH.tif") as radar1:
    #     radar = radar1.read()
    #
    # watermask = np.empty(radar.shape, dtype=rasterio.uint8)
    # watermask = np.where(radar > -17.75, 1, np.nan).reshape(radar.shape[1], radar.shape[2])
    # water_img = Image.fromarray(255 * watermask.astype('uint8'))
    # water_img.show()
    # clas = clas * watermask
    clas_dict = {}

    for i in range(clas.shape[1]):
        for j in range(clas.shape[2]):
            clas_dict[(i, j)] = clas[0, i, j]
    full_index = pd.MultiIndex.from_product([range(shape[1]), range(shape[2])], names=['i', 'j'])
    classes = pd.DataFrame({'class': pd.Series(clas_dict)}, index=full_index)
    print('*****classesshape:  ', classes.shape)
    ## Combine spectral and label data, extract training and test datasets
    data_df = classes.merge(windows, left_index=True, right_index=True, how='left')
    print('*****data_df shape:  ', data_df.shape)
    data_df[data_df <= -999] = np.nan
    data_df = data_df.dropna()
    # data_df = data_df.loc[data_df['class']>0]  # Shouldn't need to limit to a single class
    if(X_scaled.size==0):
        X_scaled = np.empty((0, windows.shape[1]), 'float64')
        y = np.empty((0), 'int')
    X = data_df[[col for col in data_df.columns if col != 'class']]
    print('*****X shape:  ', X.shape)
    scaler = StandardScaler()
    X_scaled = np.append(X_scaled, scaler.fit_transform(X.astype(np.float64)), axis=0)
    print('*****X_scaled shape:  ', X_scaled.shape)
    temp = data_df['class'].values
    y = np.append(y, temp, axis=0)
    print('*****y shape:  ', y.shape)
    return X_scaled, y, data_df, full_index, shape

train_df = get_all_concession_data(concessions)

#data_df = combine_input_landcover(allInput, landcover)
X = train_df[[col for col in train_df.columns if col != 'class']]
X_scaled = scale_data(X)
landcover = train_df['class'].values
#print(pd.Series(y).value_counts(dropna=False))
#encoder = LabelEncoder() # Could be used if we're not always dealing with the same classes
#encoder.fit(y)
#n_classes = encoder.classes_.shape[0]
#y_encoded = encoder.transform(y)

predictions = pd.DataFrame()
clas_cols = ['prob_' + str(clas) for clas in classes.values()]
probabilities_dict = {}


outtif = stack_image_input_data(classConcession)
with rio.open(outtif) as img_src:
    img = img_src.read()
    shape=img.shape
    x = gen_windows(img, pixel_window_size)
class_image = get_landcover_class_image(classConcession)
class_image=mask_water(class_image, classConcession)

y = get_classes(class_image)
#df_class = combine_input_landcover(x,y)
df_class = y.merge(x, left_index=True, right_index=True, how='left')
X_class = df_class[[col for col in df_class.columns if col != 'class']]
X_scaled_class = scale_data(X_class)
for key in classes.keys():
    probabilities_dict[key]=pd.DataFrame()
for seed in range(1,iterations+1):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, landcover, train_size=0.0200, test_size=0.1,
                                                        random_state=13*seed)

    # # =============================================================================
    # # Train and test random forest classifier
    # # =============================================================================
    clf = rfc(n_estimators=500, max_depth = 6, max_features = .3, max_leaf_nodes = 10,
              random_state=seed, oob_score = True, n_jobs = -1,
            #  class_weight = {0:0.33, 1: 0.33, 2: 0.34})
              class_weight ='balanced')
    if doGridSearch:
        param_grid = [{'max_depth': [2, 10],
                       'max_leaf_nodes': [10, 20, 50],
                       'max_features': [.25, .5, .75]}]
        grid_search = GridSearchCV(clf, param_grid, cv = 5, #scoring = 'balanced_accuracy',
                                   return_train_score = True, refit = True)

        grid_search.fit(X_train, y_train)

        randomforest_fitted_clf = grid_search.best_estimator_
    else:
        randomforest_fitted_clf = clf.fit(X_train, y_train)
    y_hat = randomforest_fitted_clf.predict(X_test)
    print('*************  RANDOM FOREST  - X_TEST  **********************')
    print(sklearn.metrics.classification_report(y_test, y_hat))
    print(sklearn.metrics.confusion_matrix(y_test, y_hat))

    y_hat = randomforest_fitted_clf.predict(X_train)
    print('*************  RANDOM FOREST  - X_TRAIN  **********************')
    print(sklearn.metrics.classification_report(y_train, y_hat))
    print(sklearn.metrics.confusion_matrix(y_train, y_hat))

    #perm_imp_rfpimp = permutation_importances(randomforest_fitted_clf, X_train, y_train, r2)
    #print(perm_imp_rfpimp)
    forest_importance_ranking(randomforest_fitted_clf)
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


    predictions[seed] = randomforest_fitted_clf.predict(X_scaled_class)
    #mytemp=randomforest_fitted_clf.predict_proba(X_scaled_class)[:,0]
    for key in classes.keys():
        probabilities_dict[key][seed] = randomforest_fitted_clf.predict_proba(X_scaled_class)[:,key]


if iterations>1:
    temp = predictions.mode(axis=1)
else:
    temp=predictions
full_index = pd.MultiIndex.from_product([range(shape[1]), range(shape[2])], names=['i', 'j'])
temp = temp.set_index(full_index)
for key in classes.keys():
    if iterations > 1:
        tempMean = probabilities_dict[key].mean(axis=1)
    else:
        tempMean = probabilities_dict[key]
    probabilities_dict[key] = pd.DataFrame(tempMean).set_index(full_index)
    probabilities_dict[key] = probabilities_dict[key].values.reshape(shape[1], shape[2])

if(iterations>1):
    df_class['predicted']=temp[0]#this should give majority class for the
else:
    df_class['predicted'] = temp
clas_df = pd.DataFrame(index = full_index)
classified = clas_df.merge(df_class['predicted'], left_index = True, right_index = True, how = 'left').sort_index()
classified = classified['predicted'].values.reshape(shape[1], shape[2])
clas_img = ((classified * 255)/2).astype('uint8')
clas_img = mask_water(clas_img,classConcession)
clas_img = Image.fromarray(clas_img)
#clas_img.show()
print('*************  RANDOM FOREST  - ACTUAL  **********************')
print(df_class.shape)
df_class[df_class <= -999] = np.nan
df_class = df_class.dropna()
print(df_class.shape)
print(sklearn.metrics.classification_report(df_class['class'], df_class['predicted']))
print(sklearn.metrics.confusion_matrix(df_class['class'], df_class['predicted']))
classified = classified[np.newaxis, :, :].astype(rio.int16)
outclas_file = base_dir + classConcession + '/sklearn_test/classified' + suffix
referencefile = base_dir + classConcession + '/' + classConcession + '_remap_2class.remapped.tif'
prob_file = base_dir + classConcession +  '/sklearn_test/prob_file' + suffix

file_list = sorted(glob.glob(referencefile))
with rio.open(file_list[0]) as src:
    height = src.height
    width = src.width
    crs = src.crs
    transform = src.transform
    dtype = rio.int16
    count = 1
    with rio.open(outclas_file, 'w', driver = 'GTiff',
                  height = height, width = width, 
                  crs = crs, dtype = dtype, 
                  count = count, transform = transform) as clas_dst:
        clas_dst.write(classified)
    with rio.open(prob_file, 'w', driver = 'GTiff',
                  height = height, width = width,
                  crs = crs, dtype = rio.float32,
                  count = len(randomforest_fitted_clf.classes_), transform = transform) as prob_dst:
        for key, value in probabilities_dict.items():
            print(key, '....')
            prob_dst.write_band(key+1, value.astype(rio.float32))
prob_dst.close()
clas_dst.close()
src.close()
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
        y_hat = pd.Series(y_hat, index = self.x_df.index)
        y_hat.name = 'y_hat'
        temp_df = self.block_df.merge(y_hat, left_index = True, right_index = True, how = 'left')
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
        pred_df = pd.DataFrame(pred_df, index = self.x_df.index, columns = clas_cols)    
        temp_df = self.block_df.merge(pred_df, left_index = True, right_index = True, how = 'left')
        probabilities = temp_df[clas_cols].to_numpy().T.reshape(len(clas_cols), self.shape[1], self.shape[2])
        probabilities = probabilities.astype(rio.float32)
        return probabilities

#clas_file = base_dir + classConcession +  '/sklearn_test/class_file.tif'
prob_file = base_dir + classConcession +  '/sklearn_test/prob_file' + suffix

# with rio.open(outtif) as src:
#     clas_dst = rio.open(clas_file, 'w', driver = 'GTiff',
#                    height = src.height, width = src.width,
#                    crs = src.crs, dtype = rio.int16,
#                    count = 1, transform = src.transform)
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
