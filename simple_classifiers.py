# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 15:12:56 2019

@author: rheil
"""
# =============================================================================
# Imports
# =============================================================================
import keras
import dirfuncs
import glob
import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
from osgeo import gdal,gdalconst
import rasterio as rio
import rasterio.warp
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




# =============================================================================
# Identify files
# =============================================================================
base_dir = dirfuncs.guess_data_dir()
#input_dir = base_dir + "in\\"
concessions = ['app_riau', 'app_jambi']
clas_file=''
referencefile=''
outvrt = '/vsimem/stacked.vrt' #/vsimem is special in-memory virtual "directory"

tifs = []#, img_file4]
classes = {2: "HCSA",
           1: "Not_HCSA",
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


def stackImageData(concession):
    input_dir = base_dir + concession + "/in/"
    print(input_dir)
    outtif = base_dir + concession + '/out/input_' + concession + '.tif'
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
                dst.write_band(i, src1.read(1).astype('float64'))
    dst.close()
    return outtif
#getData('app_kalbar_cntk')

def extractData(imagepath, concession):
    X_scaled = np.empty((0, 27), 'float64')
    y = np.empty((0), 'int')
    with rio.open(outtif) as img_src:
        img = img_src.read()

    shape = img.shape
    windows = gen_windows(img, 1)
    clas_file = base_dir + concession + '/' + concession + '*remap*.tif'
    print(clas_file)
    file_list = sorted(glob.glob(clas_file))
    ## Read classification labels
    with rio.open(file_list[0]) as clas_src:
        clas = clas_src.read()
    with rio.open(base_dir + concession + "/in/" + concession +"radar.VH.tif") as radar1:
        radar = radar1.read()

    watermask = np.empty(radar.shape, dtype=rasterio.uint8)
    watermask = np.where(radar > -17.75, 1, np.nan).reshape(radar.shape[1], radar.shape[2])
    water_img = Image.fromarray(255 * watermask.astype('uint8'))
    water_img.show()
    clas = clas * watermask
    clas_dict = {}

    for i in range(clas.shape[1]):
        for j in range(clas.shape[2]):
            clas_dict[(i, j)] = clas[0, i, j]
    full_index = pd.MultiIndex.from_product([range(shape[1]), range(shape[2])], names=['i', 'j'])
    classes = pd.DataFrame({'class': pd.Series(clas_dict)}, index=full_index)
    ## Combine spectral and label data, extract training and test datasets
    data_df = classes.merge(windows, left_index=True, right_index=True, how='left')
    data_df[data_df <= -999] = np.nan
    data_df = data_df.dropna()
    # data_df = data_df.loc[data_df['class']>0]  # Shouldn't need to limit to a single class
    X = data_df[[col for col in data_df.columns if col != 'class']]
    scaler = StandardScaler()
    X_scaled = np.append(X_scaled, scaler.fit_transform(X.astype(np.float64)), axis=0)
    temp = data_df['class'].values
    y = np.append(y, temp, axis=0)
    return X_scaled, y, data_df, full_index, shape

myX = np.empty((0,27), 'float64')
myY = np.empty((0), 'int')
for concession in concessions:
    outtif = stackImageData(concession)
    myX, myY, data, index, shape = extractData(outtif, concession)

    #print(pd.Series(y).value_counts(dropna=False))
#encoder = LabelEncoder() # Could be used if we're not always dealing with the same classes
#encoder.fit(y)
#n_classes = encoder.classes_.shape[0]
#y_encoded = encoder.transform(y)
X_train, X_test, y_train, y_test = train_test_split(myX, myY, train_size=0.06, test_size=0.02,
                                                    random_state=123)

# # =============================================================================
# # Train and test random forest classifier
# # =============================================================================
clf = rfc(n_estimators=60, max_depth = 6, max_features = .3, max_leaf_nodes = 10,
          random_state=123, oob_score = True, n_jobs = -1,
        #  class_weight = {0:0.33, 1: 0.33, 2: 0.34})
          class_weight ='balanced')
#randomforest_fitted_clf = clf.fit(X_train, y_train)
param_grid = [{'max_depth': [2, 10],
               'max_leaf_nodes': [10, 20, 50],
               'max_features': [.25, .5, .75]}]
grid_search = GridSearchCV(clf, param_grid, cv = 5, #scoring = 'balanced_accuracy',
                           return_train_score = True, refit = True)

grid_search.fit(X_train, y_train)

randomforest_fitted_clf = grid_search.best_estimator_
y_hat = randomforest_fitted_clf.predict(X_test)
print('*************  RANDOM FOREST  - X_TEST  **********************')
print(sklearn.metrics.classification_report(y_test, y_hat))
print(sklearn.metrics.confusion_matrix(y_test, y_hat))

y_hat = randomforest_fitted_clf.predict(X_train)
print('*************  RANDOM FOREST  - X_TRAIN  **********************')
print(sklearn.metrics.classification_report(y_train, y_hat))
print(sklearn.metrics.confusion_matrix(y_train, y_hat))

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
classConcession = 'app_oki'
outtif = stackImageData(classConcession)
myX, myY, data_df, full_index, shape = extractData(outtif, classConcession)
data_df['predicted'] = randomforest_fitted_clf.predict(myX)
clas_df = pd.DataFrame(index = full_index)
classified = clas_df.merge(data_df['predicted'], left_index = True, right_index = True, how = 'left').sort_index()
classified = classified['predicted'].values.reshape(shape[1], shape[2])
clas_img = ((classified * 255)/2).astype('uint8')
clas_img = Image.fromarray(clas_img)
clas_img.show()

classified = classified[np.newaxis, :, :].astype(rio.int16)
outclas_file = base_dir + classConcession + '/sklearn_test/classified.tif'
referencefile = base_dir + classConcession + '/' + classConcession + '*remap*.tif'
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

clas_file = base_dir + classConcession +  '/sklearn_test/class_file.tif'
prob_file = base_dir + classConcession +  '/sklearn_test/prob_file.tif'

with rio.open(outtif) as src:
    clas_dst = rio.open(clas_file, 'w', driver = 'GTiff', 
                   height = src.height, width = src.width, 
                   crs = src.crs, dtype = rio.int16,
                   count = 1, transform = src.transform)
    prob_dst = rio.open(prob_file, 'w', driver = 'GTiff', 
                   height = src.height, width = src.width, 
                   crs = src.crs, dtype = rio.float32, 
                   count = len(randomforest_fitted_clf.classes_), transform = src.transform)
    for ji, window in src.block_windows(1):
        block = src.read(window = window)
        if sum(sum(sum(~np.isnan(block))))>0:
            block_classifier = classify_block(block, randomforest_fitted_clf)
            classified = block_classifier.classify()
            probabilities = block_classifier.calc_probabilities()
            clas_dst.write(classified, window = window)
            prob_dst.write(probabilities, window = window)

clas_dst.close()
prob_dst.close()
