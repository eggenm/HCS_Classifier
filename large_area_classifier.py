import numpy as np
import pandas as pd
import rasterio as rio
import data.hcs_database as db
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import metrics
import data_helper as helper
import dirfuncs
import timer
import train_classsifier as trainer
import scipy.stats as stat
import csv

base_dir = dirfuncs.guess_data_dir()
shapefile = ''
year = str(2015)
sites = [
    'app_riau',
    'app_oki',
    'app_jambi'
]
sites = [
    'Bumitama_PTGemilangMakmurSubur',
    'Bumitama_PTHungarindoPersada',
    'PTAgroAndalan', 'gar_pgm',
    #'Bumitama_PTDamaiAgroSejahtera'
    #'PTMitraNusaSarana',

]
bands = ['blue_max', 'green_max', 'red_max',
         'nir_max',
         'swir1_max', 'swir2_max', 'VH', 'VV', 'VH_0', 'VV_0', 'VH_2', 'VV_2', 'EVI', 'slope'
 ]



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


def predict(X_scaled_class, rfmodel):#, predictions):
    try:
        predictions = np.zeros(X_scaled_class.shape[0], dtype=np.int8)
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

def get_map_file_name(name, id):
    outclas_file = base_dir + name + '/sklearn_test/' + name + str(id) + '_classified_by_ensemble_rf.tif'
    return outclas_file

def write_map(predicted, reference, name,i):
    outclas_file = get_map_file_name(name, i)
    src = reference
    #with rio.open(reference) as src:
    height = src.rio.height
    width = src.rio.width
    shape = src.shape
    crs = src.rio.crs
    trans = src.transform
    count = 1
    full_index = pd.MultiIndex.from_product([range(shape[1]), range(shape[2])], names=['i', 'j'])
    print(type(predicted))
    if (isinstance(predicted,(np.ndarray, pd.Series))):
        predicted = pd.DataFrame(predicted, index=full_index, dtype=np.int8)
    else:
        predicted = predicted.set_index(full_index).astype(dtype=np.int8,copy=False)

    clas_df = pd.DataFrame(index=full_index, dtype=np.int8)
    classified = clas_df.merge(predicted, left_index=True, right_index=True, how='left').sort_index()
    predicted, clas_df=False
    classified = classified.values.reshape(shape[1], shape[2])

    classified = classified[np.newaxis, :, :].astype(rio.int16)
    with rio.open(outclas_file, 'w', driver='GTiff',
                        height=height, width=width,
                        crs=crs, dtype=rio.int16,
                        count=count, transform=trans) as clas_dst:
        for ji, window in clas_dst.block_windows(1):  # or ref file here
            print('ji:  ', ji)
            print('window:  ', window)
            #  print('window.shape:  ', window.shape)
            block = classified[window.col_off:window.col_off + width,
                    window.row_off:window.row_off + height]  # .read(window=window)
            #     if sum(sum(sum(~np.isnan(block)))) > 0:
            clas_dst.write(block, window=window)
    # for ji, window in src.block_windows(1):  # or ref file here
    #     print('ji:  ', ji)
    #     print('window:  ', window)
    #     block = predicted.read(window=window)
    #     if sum(sum(sum(~np.isnan(block)))) > 0:
        #clas_dst.write(classified)#, window=window)
    clas_dst.close()
    #src.close()


def get_trained_model(scoreConcession, trainConcessions, seed):
    doGridSearch = False
    scheme = '3CLASS'#db.get_best_scheme([scoreConcession])
    band_string = '[\'blue_max\', \'green_max\', \'red_max\', \'nir_max\', \'swir1_max\', \'swir2_max\', \'VH\', \'VV\', \'VH_0\', \'VV_0\', \'VH_2\', \'VV_2\', \'EVI\', \'slope\']'
    estimators = db.get_best_number_estimators(scoreConcession, band_string)
    max_features = db.get_best_max_features(scoreConcession, band_string)
    depth = db.get_best_max_depth(scoreConcession, band_string)
    leaf_nodes = db.get_best_max_leaf_nodes(scoreConcession, band_string)
    year = str(2015)
    #bands = db.get_best_bands([scoreConcession])
   #
    # TODO take this out, just for a local test!!!!
    #print(bands)
    sample_rate = int(db.get_best_training_sample_rate(scoreConcession, band_string))

    try:
        with timer.Timer() as t:
            rf_trainer = trainer.random_forest_trainer(estimators, depth, max_features, leaf_nodes, bands, scheme)
            X_train, X_test, y_train, y_test = get_training_data(trainConcessions, bands, year, sample_rate,
                                                                 seed)
            if scheme == '3CLASS':
                y_train = helper.map_to_3class(y_train)
            rf_trainer.train_model(X_train, y_train, seed)
            X_train, X_test, y_train, y_test = get_training_data([scoreConcession], bands, year, sample_rate,
                                                                 seed)
            yhat = predict(X_test, rf_trainer)
            y_test = helper.map_to_2class(y_test)
            result = score_model(y_test, yhat)
            result['max_depth'] = rf_trainer.model.get_params()['max_depth']
            result['max_leaf_nodes']= rf_trainer.model.get_params()['max_leaf_nodes']
            result['max_features']= rf_trainer.model.get_params()['max_features']
            result['n_estimators']= rf_trainer.model.get_params()['n_estimators']
            result['samples'] = sample_rate
            result['scheme'] = scheme
    finally:
        print('Train Model Request took %.03f sec.' % t.interval)
    return rf_trainer, result


def get_training_data(sites, bands, year, sample_rate,  seed):
    train_df = helper.trim_data2(helper.get_input_data(bands, year, sites, False))
    train_df = helper.drop_no_data(train_df)
    X = train_df[[col for col in train_df.columns if (col != 'clas')]]
    X_scaled = helper.scale_data(X)
    landcover = train_df['clas'].values
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, landcover, train_size=sample_rate, test_size=0.35,
                                                        random_state=seed)
    return X_train, X_test, y_train, y_test

def score_model(y_test, yhat):
    f1 = metrics.f1_score(y_test, yhat, average='macro')
    f1_weighted = metrics.f1_score(y_test, yhat, average='weighted')
    kappa = metrics.cohen_kappa_score(y_test, yhat)
    balanced_accuracy = metrics.balanced_accuracy_score(y_test, yhat)
    roc_auc = metrics.roc_auc_score(y_test, yhat, average='weighted')
    accuracy = metrics.accuracy_score(y_test, yhat)
    return {'f1_macro':f1,'f1_weighted': f1_weighted, 'kappa': kappa, 'balanced_accuracy': balanced_accuracy,  'accuracy': accuracy, 'roc_auc_weighted': roc_auc}

def log_accuracy(result, name, id):
    csv_file = get_map_file_name(name, id) + '.csv'
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=result[0].keys())
            writer.writeheader()
            for data in result:
                writer.writerow(data)
    except IOError:
        print("I/O error")


if __name__ == "__main__":
    name = 'West_Kalimantan'
    try:
        with timer.Timer() as t:
            island = db.conncession_island_dict[name]
            ref_study_area = helper.get_reference_raster_from_shape(name, island, year)
            X_scaled_class = helper.get_large_area_input_data(ref_study_area, bands, island,
                                                              year, name)  # TODO this relies on hardcoded bands where below pulls from database

            iterations_per_site = 3
            total_predictions = iterations_per_site * len(sites)
            predictions = np.zeros((total_predictions, X_scaled_class.shape[0]), dtype=np.int8)


            k=0
            result = []
            for i, scoreConcession in enumerate(sites):
                print(scoreConcession)
                trainConcessions = list(sites)
                trainConcessions.remove(scoreConcession)
                #trained_model = get_trained_model(scoreConcession, trainConcessions, i)
                #predictions = predict(X_scaled_class, trained_model, predictions)
                #write_map(predictions, ref_study_area, name, i)


                for j in range(7*i, 7*i+iterations_per_site):
                    trained_model, scores = get_trained_model(scoreConcession, trainConcessions, j)
                    scores['oob_concessions'] = scoreConcession
                    scores['train_concessions'] = trainConcessions
                    result.append(scores)
                    log_accuracy(result,name, j)
                    predictions[k] =  predict(X_scaled_class, trained_model)#, predictions)
                    if k%3==0:
                        write_map(predictions[k], ref_study_area, name, j)
                    k=k+1
                #i = i + 1
            #predictions = predictions / number_predictions
            #predictions = np.around(predictions)
            mapId='FINAL'
            log_accuracy(result, name, mapId)
            print('predictions.shape', predictions.shape)
            myFrame = pd.DataFrame(predictions)
            temp1 = (pd.DataFrame(myFrame.T.mode(axis=1))[0]).astype(int)
            print('temp1.shape', temp1.shape)
            #print(predictions)
            #predictions = temp0.astype(int)
            #print(predictions)
            write_map(temp1.values, ref_study_area, name,mapId)
    finally:
        print('LARGE_AREA CLASSIFICATION of : ' , name , '  took %.03f sec.' % t.interval)
