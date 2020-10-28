import os
import glob
import rioxarray as rx
import dirfuncs
import glob

base_dir = dirfuncs.guess_data_dir()

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

    def get_band_by_context_year(self, band, context, year ):
        key = context+str(year)+band
        try:
            self.island_data_table[key]
        except KeyError:
            tif = os.path.join(self.base_dir, context, 'out', str(year), '*' + band + '.tif')
            file = glob.glob(tif)
            #self.island_data_table[key] = rx.open_rasterio(file[0])
            return rx.open_rasterio(file[0])
        #return self.island_data_table[key]


    def get_band_by_name_year(self, band, name,  year, context ):
        key = name + str(year) + band
        try:
            self.island_data_table[key]
        except KeyError:
            tif = self.get_input_image_path(name, year, context, band)
            file = glob.glob(tif)
            self.island_data_table[key] = rx.open_rasterio(file[0])
            #return rx.open_rasterio(file[0])
            #return False
        return self.island_data_table[key]

    def get_class_by_concession_name(self, name ):
        key = 'class' + name
        try:
            self.island_data_table[key]
        except KeyError:
            tif = os.path.join(self.base_dir, name,  name + '_all_class.remapped.tif')
            file = glob.glob(tif)
            self.island_data_table[key] = rx.open_rasterio(file[0])
            #return rx.open_rasterio(file[0])
            #return False
        return self.island_data_table[key]

    def get_fixed_class_paths(self, name, context):
        tif = os.path.join(self.base_dir, context, name,  '*.tif')
        files = glob.glob(tif)
        return files

    def get_fixed_input_image_path(self, filename, band):
        base = os.path.basename(filename)
        path = os.path.dirname(filename)
        name = base.replace('.tif', '*'+band+'*')
        name = os.path.join(path, 'out', name)
        return name

    def get_input_image_path(self, name, year, context, band):
        if (context == 'supplementary_class'):
            tif = os.path.join(self.base_dir, context, name, 'out', 'input_' + name + '_' + band + '.tif')

        else:
            tif = os.path.join(self.base_dir, name, 'out', str(year), 'input_' + name + '_' + band + '.tif')
        return tif


