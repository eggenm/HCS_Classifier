import ee
ee.Initialize()
import numpy as np

l8_band_dict =  {#'B1': 'ublue',
               'B2': 'blue_max',
       #       'B3': 'green_max',
               'B4': 'red_max',
               'B5': 'nir_max',
               'B6': 'swir1_max',
              'B7': 'swir2_max',
      #        'B10': 'tir1',
      #        'B11': 'tir2',
       #       'sr_aerosol': 'sr_aerosol'
             'EVI2': 'EVI'
              }

s2_band_dict = {
    #'B1': 'S2_ublue',
            #'B2': 'blue_max',
         #  'B3': 'green_max',
          'B4': 'red_max',
     #        'B5': 'rededge1_max',
     #      'B6': 'rededge2_max',
     #       'B7': 'rededge3_max',
  #
               'B8': 'nir_max',
        #         'B8A': 'S2_nir2_max',
     #         'B9': 'S2_vape_max',
  #
           'B11': 'swir1_max',
         'B12': 'swir2_max',
   #           'nd': 'ndvi_s2_max',
    'EVI':'EVI'
}

s2_band_dict_median = {
    # 'B1': 'S2_ublue',
     #          'B2': 'S2_blue_median',
        #        'B3': 'S2_green_median',
          #    'B4': 'S2_red_median',
             'B5': 'rededge1_median',
           'B6': 'rededge2_median',
            'B7': 'rededge3_median',
    # # ,

        #         'B8': 'S2_nir_median',
      #          'B8A': 'S2_nir2_median',
              'B9': 'S2_vape_median',
          #
              'B10': 'S2_swir1_median',
            'B11': 'S2_swir2_median',
           'B12': 'S2_swir3_median',
  #           'nd': 'ndvi_s2_median',
   'EVI2':'EVI2_s2_median'
}

s1_band_dict = {'VH': 'VH',
              'VV': 'VV',
            'VH_0': 'VH_0',
             'VV_0': 'VV_0',
           'VH_2': 'VH_2',
              'VV_2': 'VV_2',
           #   'VH-VV':'diff_VH_VV',
        #      'VH/VV': 'ratio_VH_VV'
      }

dem_band_dict = {
    'elevation':'elevation',
    'slope': 'slope',
    #'aspect':'aspect'

}
soil_band_dict = {
    'grtgroup':'grtgroup'
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


def maskClouds_L_TOA(im):
    cs = ee.Algorithms.Landsat.simpleCloudScore(im)
    cloud = cs.select('cloud').gte(20.0)
    cloud = cloud.updateMask(cloud)
    shadow = im.select('B6').lt(0.25);
    shadow = shadow.updateMask(shadow);
    return im.updateMask(mask)


def maskL8Clouds_2(image):
    qa = image.select('BQA');
    mask = qa.bitwiseAnd(ee.Number(2).pow(12).int()).eq(1).And(qa.bitwiseAnd(ee.Number(2).pow(13).int()).eq(1)).Or(
        qa.bitwiseAnd(ee.Number(2).pow(4).int()).neq(0)).And(qa.bitwiseAnd(ee.Number(2).pow(7).int()).neq(0)).And(
                                                                 qa.bitwiseAnd(ee.Number(2).pow(5).int()).neq(0)).And(
                                                                 qa.bitwiseAnd(ee.Number(2).pow(6).int()).neq(
                                                                     0)).Not();
    return image.updateMask(mask)


def maskS2clouds(image):
    qa = image.select('QA60')
    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = (1 << 10)
    cirrusBitMask = (1 << 11)
    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0).And(qa.bitwiseAnd(cirrusBitMask).eq(0));
    return image.updateMask(mask).divide(10000)

def mask_using_CDI(image):
   cdi = ee.Algorithms.Sentinel2.CDI(image);
   mask = cdi.gt(-0.3);
   return image.updateMask(mask);


def maskCloudsL5(image):
  score = ee.Algorithms.Landsat.simpleCloudScore(image).select('cloud');
  return image.updateMask(score.lte(30));


def addNDVI_l5(image):
  return image.addBands(image.normalizedDifference(['B4', 'B3']).rename('NDVI'));

def addNDVI_l8(image):
  return image.addBands(image.normalizedDifference(['B5', 'B4']).rename('NDVI'));

def unit_scale_l8_SR(image):
    return image.unitScale(0,10000)

def prep_ls8(img):
    """
    Used to map the initialized class onto an imagecollection. Will mask out clouds,
    add an ndvi band, and rename all bands.
    """
    # Mask out flagged clouds
    img = maskCloudsLandsat8(img)
    #img = maskL8Clouds_2(img)
    #img = img.unitScale(0,10000)

    # Rename bands

    # Add ndvi
   # img = img.addBands(img.normalizedDifference(['nir', 'red']))
    img = add_EVI2_l8(img)
    # Rename ndvi
    old_names = list(l8_band_dict.keys())
    new_names = list(l8_band_dict.values())
    img = img.select(old_names, new_names)
    #img = img.select(newer_names, newest_names)

    return img


def prep_sar(image_collection):
    composite = ee.Image.cat([
        image_collection.select('VH').median().rename('VH_2').unitScale(-38,4).focal_mean(6),
        image_collection.select('VV').median().rename('VV_2').unitScale(-26,13).focal_mean(6),
        image_collection.select('VH').median().rename('VH').unitScale(-38, 4).focal_mean(3),
        image_collection.select('VV').median().rename('VV').unitScale(-26, 13).focal_mean(3)   ,
        image_collection.select('VH').median().rename('VH_0').unitScale(-38, 4),
        image_collection.select('VV').median().rename('VV_0').unitScale(-26, 13)  # ,
        #(image_collection.select('VH').subtract(image_collection.select('VV'))).mean(),
     #   (image_collection.select('VH').divide(image_collection.select('VV'))).mean()
        #  sentinel1.select('VH').reduce(ee.Reducer.stdDev()).rename('VH_vari'), There are string artifacts with this operation
        # sentinel1.select('VV').reduce(ee.Reducer.stdDev()).rename('VV_vari')
    ]);
    #composite = composite.set('year', year)
    return composite


def prep_s2(img):
    # Mask out flagged clouds
    #img = maskS2clouds(img)
    img = mask_using_CDI(img)
    img = img.unitScale(0,10000)
    # Rename bands
    img = img.addBands(img.normalizedDifference(['B8', 'B4']))
    #old_names = list(s2_band_dict.keys())
   # new_names = list(s2_band_dict.values())
    # Add ndvi
    #img = add_EVI_s2(img)
    img = add_EVI2_s2(img)
   # img = img.select(old_names, new_names)
    return img

def addNDVI_s2(image):
  ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  return image.addBands(ndvi);

def add_EVI_s2(image):
    bands = {
        'BLUE':image.select('B2'),
        'RED': image.select('B4'),
        'NIR': image.select('B8')
  }
    evi = image.expression('2.5 * (NIR - RED) / ((NIR + 6.0 * RED - 7.5 * BLUE) + 1.0)', bands).rename('EVI');

    return image.addBands(evi);

def add_EVI2_s2(image):
    bands = {
        'RED': image.select('B4'),
        'NIR': image.select('B8')
  }
    evi2 = image.expression('2.4 * (NIR - RED) / (NIR + RED + 1)', bands).rename('EVI');

    return image.addBands(evi2);

def add_EVI2_l8(image):
    bands = {
        'RED': image.select('B4'),
        'NIR': image.select('B5')
  }
    evi2 = image.expression('2.4 * (NIR - RED) / (NIR + RED + 1)', bands).rename('EVI2');

    return image.addBands(evi2);

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
