import ee
ee.Initialize()
import numpy as np

l8_band_dict =  {'B1': 'ublue',
              'B2': 'blue',
              'B3': 'green',
              'B4': 'red',
              'B5': 'nir',
              'B6': 'swir1',
              'B7': 'swir2',
              'B10': 'tir1',
              'B11': 'tir2',
              'sr_aerosol': 'sr_aerosol'
#             ,'nd': 'ndvi_l8'
              }

s2_band_dict = {
    # 'B1': 'S2_ublue',
    #           'B2': 'S2_blue',
    #           'B3': 'S2_green',
    #           'B4': 'S2_red',
    #         'B5': 'rededge1'#,
    #         'B6': 'rededge2',
    #         'B7': 'rededge3'
    # ,
    #            'B8': 'S2_nir',
    #            'B8A': 'S2_nir2',
    #            'B9': 'S2_vape'
    #,
               'B10': 'S2_swir1',
             'B11': 'S2_swir2',
             'B12': 'S2_swir3',
              'nd': 'ndvi_s2'
}

s1_band_dict = {'VH': 'VH',
              'VV': 'VV'#,
           #   'VH-VV':'diff_VH_VV',
        #      'VH/VV': 'ratio_VH_VV'
      }

def maskCloudsLandsat8(image):
    # Bits 3 and 5 are cloud shadow and cloud, respectively.
    cloudShadowBitMask = (1 << 3)
    cloudsBitMask = (1 << 5)
    # Get the pixel QA band.
    qa = image.select('pixel_qa')
    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(qa.bitwiseAnd(cloudsBitMask).eq(0))
    return image.updateMask(mask)

def maskS2clouds(image):
    qa = image.select('QA60')
    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = (1 << 10)
    cirrusBitMask = (1 << 11)
    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0));
    return image.updateMask(mask).divide(10000)


def prep_ls8(img):
    """
    Used to map the initialized class onto an imagecollection. Will mask out clouds,
    add an ndvi band, and rename all bands.
    """
    # Mask out flagged clouds
    img = maskCloudsLandsat8(img)

    # Rename bands
    old_names = list(l8_band_dict.keys())
    new_names = list(l8_band_dict.values())
    img = img.select(old_names, new_names)
    # Add ndvi
    img = img.addBands(img.normalizedDifference(['nir', 'red']))

    # Rename ndvi
    newer_names = new_names.copy()
    newest_names = new_names.copy()
    newer_names.append('nd')
    newest_names.append('ndvi_l8')
    img = img.select(newer_names, newest_names)

    return img


def prep_sar(image_collection):
    composite = ee.Image.cat([
        image_collection.select('VH').mean(),
        image_collection.select('VV').mean()#,
        #(image_collection.select('VH').subtract(image_collection.select('VV'))).mean(),
     #   (image_collection.select('VH').divide(image_collection.select('VV'))).mean()
        #  sentinel1.select('VH').reduce(ee.Reducer.stdDev()).rename('VH_vari'), There are string artifacts with this operation
        # sentinel1.select('VV').reduce(ee.Reducer.stdDev()).rename('VV_vari')
    ]).focal_median();
    #composite = composite.set('year', year)
    return composite


def prep_s2(img):
    # Mask out flagged clouds
    img = maskS2clouds(img)
    # Rename bands
    img = img.addBands(img.normalizedDifference(['B8', 'B4']))
    old_names = list(s2_band_dict.keys())
    new_names = list(s2_band_dict.values())
    img = img.select(old_names, new_names)
    # Add ndvi

    return img


def add_ndvi(img, keys, values, platform):
    print(platform)
    # Add ndvi
    # band_names=
    newer_names = keys.copy()
    newest_names = values.copy()
    newest_names.replace('nir', 'nir' + platform)
    newest_names.replace('red', 'red' + platform)
    img = img.addBands(img.normalizedDifference(['nir', 'red']))
    # Rename ndvi
    newer_names.append('nd')
    newest_names.append('ndvi' + platform)
    newest_names.replace('nir', 'nir' + platform)
    newest_names.replace('red', 'red' + platform)

    #    newer_names = list(['nd']) + list(band_names)
    #    newer_names.extend(band_names)
    #    newest_names = list(['ndvi']) + list(band_names)
    #    newest_names.extend(band_names)
    print(newest_names)
    img = img.select(newer_names, newest_names)
    return img
