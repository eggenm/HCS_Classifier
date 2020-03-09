import numpy as np
import pandas as pd
import rasterio as rio
import train_classsifier as trainer
import data_helper as helper
import dirfuncs

base_dir = dirfuncs.guess_data_dir()
shapefile = ''
island='Sumatra'
year=str(2015)
sites = [
'app_riau',
'app_oki',
'app_jambi'
    ]
bands = ['blue_max', 'green_max', 'red_max', 'nir_max', 'swir1_max', 'swir2_max', 'VH', 'VV', 'EVI']

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



def get_large_area_input_data():
    df_class = helper.get_input_data_by_shape(bands, island, year, shapefile, True)
    X_class = df_class[[col for col in df_class.columns if (col != 'clas')]]
    X_scaled_class = helper.scale_data(X_class)
    print('X_scaled_class.shape:  ', X_scaled_class.shape)

    X_class = 0


def predict(X_scaled_class, model):
    end = X_scaled_class.shape[0]
    step = 1000000
    for i in range(0, end, step):
        y = min(i + step, end)
        print(i, y)
        block = X_scaled_class[i:y, :]
        print('block.shape: ', block.shape)
        if (model.scheme == 'ALL'):
            predictions[i:y] = predictions[i:y] + helper.map_to_2class(model.predict(block))
            # test= helper.map_to_2class(randomforest_fitted_clf.predict(block))
        else:
            predictions[i:y] = predictions[i:y] + helper.map_3_to_2class(model.predict(block))
    return  predictions

def write_map(predicted, reference, name):
    outclas_file = base_dir + name + '/sklearn_test/classified_by_ensemble_rf.tif'
    with rio.open(reference) as src:
        height = src.height
        width = src.width
        shape = src.shape
        crs = src.crs
        transform = src.transform
        dtype = rio.int16
        count = 1
        full_index = pd.MultiIndex.from_product([range(shape[0]), range(shape[1])], names=['i', 'j'])
        predicted = predicted.set_index(full_index)
        predicted = predicted.values.reshape(shape[0], shape[1])
        predicted = predicted[np.newaxis, :, :].astype(rio.int16)
        clas_dst = rio.open(outclas_file, 'w', driver='GTiff',
                            height=height, width=width,
                            crs=crs, dtype=dtype,
                            count=count, transform=transform)
        for ji, window in predicted.block_windows(1):  #or ref file here
            print('ji:  ', ji)
            print('window:  ', window)
            block = X_scaled_class.read(window=window)
            if sum(sum(sum(~np.isnan(block)))) > 0:
                clas_dst.write(predicted, window=window)
    clas_dst.close()
    src.close()


if __name__ == "__main__":
    name = 'app_muba'
    ref_study_area = helper.get_reference_raster_from_shape(name, island, year)
    X_scaled_class = helper.get_large_area_input_data(ref_study_area, bands, island, year)
    number_predictions = 2 * len(sites)
    predictions = np.zeros(( X_scaled_class.shape[0]))
    i=1
    for scoreConcession in sites:
        print(scoreConcession)
        trainConcessions = list(sites)
        trainConcessions.remove(scoreConcession)
        trained_model = trainer.get_trained_model(scoreConcession,trainConcessions, i)
        i=i+1
        predictions = predict(X_scaled_class, trained_model)
        trained_model = trainer.get_trained_model(scoreConcession, sites, i)
        i=i+1
        predictions = predict(X_scaled_class, trained_model)
    predictions = int(round(float(predictions)/number_predictions, 0))
    write_map(predictions, ref_study_area, name)
