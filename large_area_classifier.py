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
import glob
import rioxarray as rx
import sampler

base_dir = dirfuncs.guess_data_dir()
shapefile = ''
#year = str(2017)

sites = [

'Bumitama_PTDamaiAgroSejahtera',
'Bumitama_PTHungarindoPersada',
'PTMitraNusaSarana',
'makmur_abadi',
'sawit_perdana',
'aneka_sawit',
'PTMentariPratama',
'PTSukajadiSawitMekar',
'PTLabontaraEkaKarsa',
    'forest', 'impervious', 'coconut', 'pulp_and_paper', 'water', 'oil_palm'
    #,

    #'app_jambi'
 #   'impervious',
 #   'forest'
#
]
bands = [#'blue_max', 'green_max', 'red_max',
         'nir_max',
        'swir1_max', 'VH_2', 'VV_2', 'EVI','swir2_max',  'slope',  'VH_0', 'VV_0' #'VH', 'VV', 'VH_0', 'VV_0', 'VH_2', 'VV_2', 'EVI', 'slope'
 ]

my_sampler = sampler.Sampler()
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

        predictions = np.zeros(X_scaled_class.shape[0], dtype=rio.int16)
        with timer.Timer() as t:
            end = X_scaled_class.shape[0]
            step = 1000000
            for i in range(0, end, step):
                y = min(i + step, end)
                block = X_scaled_class.values[i:y, :]
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
    dtype = rio.int16
    full_index = pd.MultiIndex.from_product([range(shape[1]), range(shape[2])], names=['i', 'j'])
    # predicted = predicted.set_index(full_index)
    predicted = pd.DataFrame(predicted, index=full_index)
    clas_df = pd.DataFrame(index=full_index)
    classified = clas_df.merge(predicted, left_index=True, right_index=True, how='left').sort_index()

    classified = classified.values.reshape(shape[1], shape[2])
    #classified = classified.T.values.reshape(shape[1], shape[2])
    #classified = classified.T.values.reshape( shape[2], shape[1])
    #classified = classified.values.reshape(shape[2], shape[1])
    classified = classified[np.newaxis, :, :].astype(rio.int16)
    with rio.open(outclas_file, 'w', driver='GTiff',
                  height=height, width=width,
                  crs=crs, dtype=dtype,
                  count=count, transform=trans) as clas_dst:
        for ji, window in clas_dst.block_windows(1):  # or ref file here
            #  print('window.shape:  ', window.shape)
            block = classified[window.col_off:window.col_off + window.width,
                    window.row_off:window.row_off + window.height]  # .read(window=window)
            #     if sum(sum(sum(~np.isnan(block)))) > 0:
            clas_dst.write(block, window=window)
        #clas_dst.write(classified)
    # for ji, window in src.block_windows(1):  # or ref file here
    #     print('ji:  ', ji)
    #     print('window:  ', window)
    #     block = predicted.read(window=window)
    #     if sum(sum(sum(~np.isnan(block)))) > 0:
        #clas_dst.write(classified)#, window=window)
    clas_dst.close()
    #src.close()


def get_trained_model(scoreConcession, trainConcessions, seed, override_bands = None):
    doGridSearch = False
    scheme = '3CLASS'#db.get_best_scheme([scoreConcession])
    #band_string = '[\'blue_max\', \'green_max\', \'red_max\', \'nir_max\', \'swir1_max\', \'swir2_max\', \'VH\', \'VV\', \'VH_0\', \'VV_0\', \'VH_2\', \'VV_2\', \'EVI\', \'slope\']'
    estimators = db.get_best_number_estimators(scoreConcession)
    max_features = db.get_best_max_features(scoreConcession)
    depth = 8#db.get_best_max_depth([scoreConcession])
    leaf_nodes = db.get_best_max_leaf_nodes(scoreConcession)
    metric = 'F1'#db.get_best_metric([scoreConcession])
    year = str(2017)
    if(not override_bands):
        bands = db.get_best_bands([scoreConcession])
    else:
        bands = override_bands
    print(bands)
    sample_rate = 550#int(db.get_best_training_sample_rate([scoreConcession]))

    try:
        with timer.Timer() as t:
            rf_trainer = trainer.random_forest_trainer(estimators, depth, max_features, leaf_nodes, bands, scheme, metric)
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
            result['calibration_metric'] = metric
            result['bands'] = bands
    finally:
        print('Train Model Request took %.03f sec.' % t.interval)
    return rf_trainer, result


def get_training_data(sites, bands, year, sample_rate,  seed):
    train_dict = helper.get_input_data(bands, year, sites, False)
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    y_train = np.empty(0)
    y_test = np.empty(0)
    sample_sizes_dict = my_sampler.get_sample_rate_by_type(sample_rate, sites)
    for site in sites:
        train_df = helper.trim_data2(train_dict[site])
        train_df = helper.drop_no_data(train_df)
        X = train_df[[col for col in train_df.columns if (col != 'clas')]]
        #X_scaled = helper.scale_data(X)
        landcover = train_df['clas'].values
        train_sample = int(sample_sizes_dict[db.data_context_dict[site]][0])
        test_sample =sample_sizes_dict[db.data_context_dict[site]][1]
        X_train_site, X_test_site, y_train_site, y_test_site = train_test_split(X, landcover, train_size=train_sample, test_size=test_sample,
                                                            random_state=seed)
        X_train=pd.concat([X_train, X_train_site], ignore_index=True)
        X_test=pd.concat([X_test, X_test_site], ignore_index=True)
        y_train = np.concatenate([y_train, y_train_site])
        y_test = np.concatenate([y_test, y_test_site])

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
        csvfile.close()
    except IOError:
        print("%%%%%%%%%%%%%%%%%%        I/O error.  Log file not written:  ", csvfile,  '            %%%%%%%%%%%%%%%%%%%%%%')


if __name__ == "__main__":
    name = 'West_Kalimantan'
    for year in [2017,2018,2019]:
        try:
            with timer.Timer() as t:
                island = db.data_context_dict[name]
                tif = base_dir + name + '/out/' + year + '/input_' + name + '_' + bands[0] + '.tif'
                #tif = base_dir + name + '/out/' + year + '/' + bands[0] + '.tif'
                try:
                     file_list = sorted(glob.glob(tif))
                     ref_study_area = rx.open_rasterio(file_list[0])
                except:
                    ref_study_area = helper.get_reference_raster_from_shape(name, island, year)
                # TODO this relies on hardcoded bands where below pulls from database
                X_scaled_class = helper.get_large_area_input_data(ref_study_area, bands, island, year, name)
                iterations_per_site = 1
                total_predictions = 9 #iterations_per_site * len(sites)
                #predictions = np.zeros((total_predictions, X_scaled_class.shape[0]), dtype=np.int8)
                predictions = np.zeros(X_scaled_class.shape[0])
                #write_map(predictions, ref_study_area, name, "SWIRTEST")

                #x = False


                k=0
                result = []
                for i, scoreConcession in enumerate(sites):
                    if(db.data_context_dict[scoreConcession]=='supplementary_class'):
                        continue
                    print(scoreConcession)
                    trainConcessions = list(sites)
                    trainConcessions.remove(scoreConcession)
                    #trained_model = get_trained_model(scoreConcession, trainConcessions, i)
                    #predictions = predict(X_scaled_class, trained_model, predictions)
                    #write_map(predictions, ref_study_area, name, i)


                    for j in range(7*i, 7*i+iterations_per_site):
                        trained_model, scores = get_trained_model(scoreConcession, trainConcessions, j, bands)
                        scores['oob_concessions'] = scoreConcession
                        scores['train_concessions'] = trainConcessions

                        result.append(scores)
                        log_accuracy(result,name, j)
                        print('**********  BANDS:  ', scores['bands'], '   ************')
                        X_scaled_class = helper.get_large_area_input_data(ref_study_area, scores['bands'], island,
                                                                          year, name)
                        predictions  =  predictions + predict(X_scaled_class, trained_model)#, predictions)
                        if k%9==0:
                            write_map((np.around(predictions/(k+1))).astype(rio.int16), ref_study_area, name, j)
                        k=k+1
                        if('slope' not in scores['bands']):
                            scores['bands'].append('slope')
                            bands = scores['bands']
                            print('**********  BANDS with slope:  ',bands, '   ************')
                            trained_model, scores = get_trained_model(scoreConcession, trainConcessions, j+100, bands)
                            scores['oob_concessions'] = scoreConcession
                            scores['train_concessions'] = trainConcessions
                            result.append(scores)
                            log_accuracy(result, name, j+100)
                            print('**********  BANDS - slope:  ', scores['bands'], '   ************')
                            X_scaled_class = helper.get_large_area_input_data(ref_study_area, scores['bands'], island,
                                                                              year, name)
                            predictions = predictions + predict(X_scaled_class, trained_model)  # , predictions)
                            if k % 9 == 0:
                                write_map((np.around(predictions / (k + 1))).astype(rio.int16), ref_study_area, name, j+100)
                            k = k + 1

                    #i = i + 1
                predictions = predictions.astype(np.float32)
                predictions = predictions / k
                predictions = np.around(predictions).astype(rio.int16)
                mapId='FINAL'
                log_accuracy(result, name, mapId)
                print('predictions.shape', predictions.shape)
                #myFrame = pd.DataFrame(predictions)
                #temp1 = (pd.DataFrame(myFrame.T.mode(axis=1))[0]).astype(int)
                #print('temp1.shape', temp1.shape)
                #print(predictions)
                #predictions = temp0.astype(int)
                #print(predictions)
                write_map(predictions, ref_study_area, name,mapId)
        finally:
            print('LARGE_AREA CLASSIFICATION of : ' , name , '  took %.03f sec.' % t.interval)
