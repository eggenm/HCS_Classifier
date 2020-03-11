import numpy as np
import pandas as pd
import rasterio as rio
import data.hcs_database as db
from sklearn.model_selection import train_test_split, GridSearchCV
import data_helper as helper
import dirfuncs
import timer
import train_classsifier as trainer

base_dir = dirfuncs.guess_data_dir()
shapefile = ''
island = 'Sumatra'
year = str(2015)
sites = [
    'app_riau',
    'app_oki',
    'app_jambi'
]
bands = [  # 'blue_max',
    #  'red_max', 'nir_max',
    'swir1_max',  # 'swir2_max',
    'VH']


#    , 'VV', 'EVI']

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
        print('self.shape:  ', self.shape)
        block = block.reshape((self.shape[0], self.shape[1] * self.shape[2])).T
        print('self.shape2:  ', block.shape)
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
        clas_cols = ['prob_' + str(clas) for clas in self.fitted_clf.classes_]
        pred_df = self.fitted_clf.predict_proba(self.x_df)
        pred_df = pd.DataFrame(pred_df, index=self.x_df.index, columns=clas_cols)
        temp_df = self.block_df.merge(pred_df, left_index=True, right_index=True, how='left')
        probabilities = temp_df[clas_cols].to_numpy().T.reshape(len(clas_cols), self.shape[1], self.shape[2])
        probabilities = probabilities.astype(rio.float32)
        return probabilities


def predict(X_scaled_class, rfmodel, predictions):
    try:
        with timer.Timer() as t:
            end = X_scaled_class.shape[0]
            step = 1000000
            for i in range(0, end, step):
                y = min(i + step, end)
                print(i, y)
                block = X_scaled_class[i:y, :]
                print('block.shape: ', block.shape)
                if (rfmodel.scheme == 'ALL'):
                    predictions[i:y] = predictions[i:y] + helper.map_to_2class(rfmodel.model.predict(block))
                    # test= helper.map_to_2class(randomforest_fitted_clf.predict(block))
                else:
                    predictions[i:y] = predictions[i:y] + helper.map_3_to_2class(rfmodel.model.predict(block))
    finally:
        print('Block Predict Request took %.03f sec.' % t.interval)
    return predictions


def write_map(predicted, reference, name):
    outclas_file = base_dir + name + '/sklearn_test/' +name+'_classified_by_ensemble_rf.tif'
    src = reference
    #with rio.open(reference) as src:
    height = src.rio.height
    width = src.rio.width
    shape = src.rio.shape
    crs = src.rio.crs
    trans = src.transform
    count = 1
    full_index = pd.MultiIndex.from_product([range(shape[0]), range(shape[1])], names=['i', 'j'])
    predicted = pd.DataFrame(predicted, index=full_index)
    #predicted = predicted.set_index(full_index)
    predicted = predicted.values.reshape(shape[0], shape[1])
    predicted = predicted[np.newaxis, :, :].astype(rio.int16)
    clas_dst = rio.open(outclas_file, 'w', driver='GTiff',
                        height=height, width=width,
                        crs=crs, dtype=rio.int16,
                        count=count, transform=trans)
    # for ji, window in src.block_windows(1):  # or ref file here
    #     print('ji:  ', ji)
    #     print('window:  ', window)
    #     block = predicted.read(window=window)
    #     if sum(sum(sum(~np.isnan(block)))) > 0:
    clas_dst.write(predicted)#, window=window)
    clas_dst.close()
    #src.close()


def get_trained_model(scoreConcession, trainConcessions, seed):
    doGridSearch = False
    scheme = db.get_best_scheme([scoreConcession])
    estimators = db.get_best_number_estimators([scoreConcession])
    max_features = db.get_best_max_features([scoreConcession])
    depth = db.get_best_max_depth([scoreConcession])
    leaf_nodes = db.get_best_max_leaf_nodes([scoreConcession])
    island = 'Sumatra'
    year = str(2015)
    bands = db.get_best_bands([scoreConcession])
    bands = ['swir1_max',
             'VH']  # TODO take this out, just for a local test!!!!
    print(bands)
    sample_rate = db.get_best_training_sample_rate([scoreConcession])

    try:
        with timer.Timer() as t:
            rf_trainer = trainer.random_forest_trainer(estimators, depth, max_features, leaf_nodes, bands, scheme)
            X_train, X_test, y_train, y_test = get_training_data(trainConcessions, bands, year, sample_rate, island,
                                                                 seed)
            if scheme == '3CLASS':
                y_train = helper.map_to_3class(y_train)
            rf_trainer.train_model(X_train, y_train, seed)
    finally:
        print('Train Model Request took %.03f sec.' % t.interval)
    return rf_trainer


def get_training_data(sites, bands, year, sample_rate, island, seed):
    train_df = helper.trim_data2(helper.get_input_data(bands, island, year, sites, False))
    train_df = helper.drop_no_data(train_df)
    X = train_df[[col for col in train_df.columns if (col != 'clas')]]
    X_scaled = helper.scale_data(X)
    landcover = train_df['clas'].values
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, landcover, train_size=sample_rate, test_size=0.1,
                                                        random_state=13 * seed)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    name = 'app_muba'
    ref_study_area = helper.get_reference_raster_from_shape(name, island, year)
    X_scaled_class = helper.get_large_area_input_data(ref_study_area, bands, island,
                                                      year)  # TODO this relies on hardcoded bands where below pulls from database
    number_predictions = 2 * len(sites)
    predictions = np.zeros((X_scaled_class.shape[0]))

    i = 1
    for scoreConcession in sites:
        print(scoreConcession)
        trainConcessions = list(sites)
        trainConcessions.remove(scoreConcession)
        trained_model = get_trained_model(scoreConcession, trainConcessions, i)
        predictions = predict(X_scaled_class, trained_model, predictions)
        i = i + 1
        trained_model = get_trained_model(scoreConcession, sites, i)

        predictions =  predict(X_scaled_class, trained_model, predictions)
        i = i + 1
    predictions = predictions / number_predictions
    predictions = np.around(predictions)
    print(predictions)
    predictions = predictions.astype(int)
    print(predictions)
    write_map(predictions, ref_study_area, name)
