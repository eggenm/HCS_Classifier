import ee

l8_band_dict =  {#'B1': 'ublue',
               'B2': 'blue_max',
          #    'B3': 'green_max',
               'B4': 'red_max',
               'B5': 'nir_max',
               'B6': 'swir1_max',
              'B7': 'swir2_max',
      #        'B10': 'tir1',
      #        'B11': 'tir2',
       #       'sr_aerosol': 'sr_aerosol'
             'EVI2': 'EVI',
            'brightness':   'brightness',
            'greenness':  'greenness',
            'wetness':'wetness',
            # 'fourth': 'fourth',
            # 'fifth': 'fifth',
            # 'sixth': 'sixth'
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
def maskCloudsLandsat8(image):
    # Bits 3 and 5 are cloud shadow and cloud, respectively.
    cloudShadowBitMask = (1 << 3)
    cloudsBitMask = (1 << 5)
    # Get the pixel QA band.
    qa = image.select('pixel_qa')
    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(qa.bitwiseAnd(cloudsBitMask).eq(0))
    return image.updateMask(mask)


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

def add_tassle_cap_l8(image):
    b = image.select("B2", "B3", "B4", "B5", "B6", "B7");
   #Coefficients are only for Landsat 8 TOA
    brightness_coefficents = ee.Image([0.3029, 0.2786, 0.4733, 0.5599, 0.508, 0.1872])
    greenness_coefficents = ee.Image([-0.2941, -0.243, -0.5424, 0.7276, 0.0713, -0.1608]);
    wetness_coefficents = ee.Image([0.1511, 0.1973, 0.3283, 0.3407, -0.7117, -0.4559]);
    fourth_coefficents = ee.Image([-0.8239, 0.0849, 0.4396, -0.058, 0.2013, -0.2773]);
    fifth_coefficents = ee.Image([-0.3294, 0.0557, 0.1056, 0.1855, -0.4349, 0.8085]);
    sixth_coefficents = ee.Image([0.1079, -0.9023, 0.4119, 0.0575, -0.0259, 0.0252]);

    brightness = image.expression(
        '(B * BRIGHTNESS)',
        {
            'B': b,
            'BRIGHTNESS': brightness_coefficents
        }
    );
    greenness = image.expression(
        '(B * GREENNESS)',
        {
            'B': b,
            'GREENNESS': greenness_coefficents
        }
    );
    wetness = image.expression(
        '(B * WETNESS)',
        {
            'B': b,
            'WETNESS': wetness_coefficents
        }
    );
    # fourth = image.expression(
    #     '(B * FOURTH)',
    #     {
    #         'B': b,
    #         'FOURTH': fourth_coefficents
    #     }
    # );
    # fifth = image.expression(
    #     '(B * FIFTH)',
    #     {
    #         'B': b,
    #         'FIFTH': fifth_coefficents
    #     }
    # );
    #
    # sixth = image.expression(
    #     '(B * SIXTH)',
    #     {
    #         'B': b,
    #         'SIXTH': sixth_coefficents
    #     }
    # );
    brightness = brightness.reduce(ee.call("Reducer.sum"));
    greenness = greenness.reduce(ee.call("Reducer.sum"));
    wetness = wetness.reduce(ee.call("Reducer.sum"));
    # fourth = fourth.reduce(ee.call("Reducer.sum"));
    # fifth = fifth.reduce(ee.call("Reducer.sum"));
    # sixth = sixth.reduce(ee.call("Reducer.sum"));
    tasseled_cap = ee.Image(brightness).addBands(greenness).addBands(wetness).rename('brightness', 'greenness', 'wetness')
    return image.addBands(tasseled_cap);

def add_cloud_score_mask(image):
  cloud_param = 25
  scored = ee.Algorithms.Landsat.simpleCloudScore(image);
  mask = scored.select(['cloud']).lte(cloud_param);
  masked = image.updateMask(mask);
  return masked.set('SENSOR_ID', 'OLI_TIRS');


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
