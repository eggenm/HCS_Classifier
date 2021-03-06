from WBT.whitebox_tools import WhiteboxTools
import dirfuncs as dir
import os
import glob
import rioxarray as rx
import model_evaluator as eval
import numpy as np
import data_helper

def reproject_match_input_band(concession_tif, result_tif):
    image2 = result_tif # rx.open_rasterio(bounding_raster)
    print('image2.shape:  ', image2.shape)
    # plt.figure()
    # image2.plot()
    # plt.show()
    image3 = concession_tif
    print('image3.shape:  ', image3.shape)
    if(image3.dtype=='float64'):
        image3.data  = np.float32(image3)
    # plt.figure()
    # image3.plot()
    # plt.show()
    image3 = image2.rio.reproject_match(image3)
    print('image2.shape:  ', image2.shape)
    print('image3.shape:  ', image3.shape)
   # plt.figure()
   # destination.plot(robust=True)
  #  plt.show()

    image2=False
    #image3=False
    return image3

if __name__ == "__main__":
     wbt = WhiteboxTools()
     wbt.set_verbose_mode(True)
     name = 'Kalimantan'
     file1 = "KalimantanFINAL_classified_by_ensemble_rf.tif"
     year = 2017
     base_dir = dir.guess_data_dir()
     wbt.work_dir = os.path.join(base_dir, name, 'sklearn_test', str(year), 'test_temp', 'wkdir')
     # wbt.majority_filter(file, "Kalimantan2018_Final_smooth3x3.tif", filterx=3, filtery=3)
     #
     base = os.path.join(base_dir, name, 'sklearn_test', str(year), 'test_temp', 'Kalimantan2017_Final_100m.tif')
     out1= "resample2017.tif"
     out2 = "round2017.tif"
     wbt.resample( file1, out1, "cc",  base=base )
     wbt.round(
         out1,
         out2
     )

     wbt.majority_filter(out2, "RES_RND_SMTH_Kalimantan2017_Final3x3.tif", filterx=3, filtery=3)

     # wbt.resample(
     #     "Kalimantan2019_Final_smooth3x3.tif",
     #     "Kalimantan2019_Final_100m.tif",
     #     cell_size=0.000898315284119522,
     #     method="cc"
     # )
     # year = 2018
     # base_dir = dir.guess_data_dir()
     # wbt.work_dir = os.path.join(base_dir, name, 'sklearn_test', str(year))
     # wbt.resample(
     #        "Kalimantan2018_Final_smooth3x3.tif",
     #        "Kalimantan2018_Final_100m.tif",
     #        cell_size=0.000898315284119522,
     #         method="cc")

    #read all class remap
     # array = np.zeros((2))
     # tif_hat = os.path.join(base_dir, name, 'sklearn_test', 'Jambi_Final_smooth3x3_Clip.tif')
     # tif_true = os.path.join(base_dir, 'app_jambi', '*jambi_all_class.remapped.tif')
     # file = glob.glob(tif_true)
     # # self.island_data_table[key] = rx.open_rasterio(file[0])
     # y = rx.open_rasterio(file[0])
     # file = glob.glob(tif_hat)
     # yhat = rx.open_rasterio(file[0])
     # yhat = reproject_match_input_band(y, yhat)
     # array = np.zeros((2, y.shape[1], y.shape[2]))
     # array[0] = np.asarray(y[0])
     # array[1] = np.asarray(yhat[0])
     #
     # x = data_helper.gen_windows2(array)
     # x = data_helper.drop_no_data(x)
  #   eval.show_results(data_helper.map_to_2class(x[0]), x[1])
    #MAP IT TO 2 CLASSES
    #READ RESULT FILE
    #CHECK DIMENSIONS
    #CALL show results
