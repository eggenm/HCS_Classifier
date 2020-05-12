import os
import glob
import rioxarray as rx
import dirfuncs

class Imagery_Cache:
    __instance = None

    @staticmethod
    def getInstance():
        """ Static access method. """
        if Imagery_Cache.__instance == None:
            Imagery_Cache()
        return Imagery_Cache.__instance

    def __init__(self):
        """ Virtually private constructor. """
        self.base_dir = dirfuncs.guess_data_dir()
        self.island_data_table = {}
        if Imagery_Cache.__instance != None:
            raise Exception("This class is a singleton!")
        else:
            Imagery_Cache.__instance = self

    def get_band_by_island_year(self, band, island, year ):
        key = island+str(year)+band
        try:
            self.island_data_table[key]
        except KeyError:
            tif = os.path.join(self.base_dir, island, 'out', str(year), band + '.tif')
            file = glob.glob(tif)
            #self.island_data_table[key] = rx.open_rasterio(file[0])
            return rx.open_rasterio(file[0])
        #return self.island_data_table[key]


    def get_band_by_concession_name_year(self, band, name, island, year ):
        key = name + str(year) + band
        try:
            self.island_data_table[key]
        except KeyError:
            return False
        return self.island_data_table[key]

