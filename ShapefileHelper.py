import numpy as np
#############################
#
#  Imports are ordered this way for some unknown but necessary reason.
#  Unused pyproj comes before geopandas comes before osgeo
#  This was how I got past errors that were totally baffling, probably specific to my environment
#
#############################
import pyproj
import geopandas as gpd
from osgeo import ogr, gdal, osr
import os
import rasterio as rio
from shapely.geometry import box
import json as js
from shapely.geometry import shape, GeometryCollection
from fiona.crs import from_epsg
from rasterio import features as rioft
import imagery_data
import dirfuncs
import satellite_image_operations as sat_ops

image_cache = imagery_data.Imagery_Cache.getInstance()

supplemental_class_codes = {
   'impervious':23,
    'forest':7,
    'oil_palm':11,
    'water': 16,
  ##  'pulp_and_paper': 21,
   #  'coconut': 13,

}

def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    return [js.loads(gdf.to_json())['features'][0]['geometry']]


def ingest_kml_fixed_classes():
    #input = 'C:\\Users\\ME\\Dropbox\\HCSproject\\data\\PoC\\supplementary_class\\impervious\\doc.kml'
    imageSumatra = image_cache.get_band_by_context_year('nir_max', 'Sumatra', 2019)
    imageKalimantan = image_cache.get_band_by_context_year('nir_max', 'Kalimantan', 2019)
    imagePapua = image_cache.get_band_by_context_year('nir_max', 'Papua', 2019)

    for landcover in supplemental_class_codes.keys():
        print(landcover)
        input = os.path.join(dirfuncs.guess_data_dir(), 'supplementary_class', landcover,'doc.kml')
        srcDS = gdal.OpenEx(input)
        output = os.path.join(dirfuncs.guess_data_dir(), 'supplementary_class', landcover, landcover + '.json')
        #ds = gdal.VectorTranslate(output, srcDS, format='GeoJSON')
        #file = glob.glob(output)
        print(output)
        with open(output) as f:
                data = f.read()
                #TODO on windows you may get a parsing error that does not make sense but it has to do with EOL characters
                jsonload = js.loads(data)
                features = jsonload["features"]
                print(features)

        # NOTE: buffer(0) is a trick for fixing scenarios where polygons have overlapping coordinates
        #temp = GeometryCollection([shape(feature["geometry"]).buffer(0) for feature in features])

        #TODO get the year from the json or doc.kml

        #my_dict = sat_ops.s1_band_dict.copy()
        my_dict = sat_ops.l8_band_dict.copy()
        #my_dict.update(sat_ops.s2_band_dict)
        #my_dict.update(sat_ops.dem_band_dict)
        bands = my_dict.values()
        print(bands)



        shapes = ((shape(feature["geometry"]).buffer(0), (feature['properties']['Description']),
                   feature['properties']['Name']) for feature in features)
        for geom in shapes:
            print(geom)
            get_reference = True
            xmin, ymin, xmax, ymax = geom[0].bounds
            year_list = geom[1].split(sep=',')
            year_list = map(int, year_list)
            year_list = list(year_list)
            year_list.sort(reverse=True)
            my_year = year_list[0]
            if my_year == 2018: my_year=2019
            if my_year == 2016: my_year=2017 # TODO these only appliy because we are just using landsat 8 data aggregated in 2 year chuncks
            name = geom[2]
            for band in bands:
                if get_reference:
                    bbox = box(xmin, ymin, xmax, ymax)
                    geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=from_epsg(4326))
                    geo = geo.to_crs(crs=from_epsg(4326))
                    coords = getFeatures(geo)
                    if 'kal' in name:
                        out_img = imageKalimantan.rio.clip(coords, imageKalimantan.rio.crs)
                        island = 'Kalimantan'
                    elif 'sum' in name:
                        out_img = imageSumatra.rio.clip(coords, imageSumatra.rio.crs)
                        island = 'Sumatra'
                    elif 'pap' in name:
                        out_img = imagePapua.rio.clip(coords, imagePapua.rio.crs)
                        island = 'Papua'
                    else:
                        raise RuntimeError
                    # meta = out_img.meta.copy()
                    print(island)
                    trans = out_img.transform
                    crs = out_img.rio.crs
                    height = out_img.rio.height
                    width = out_img.rio.width
                    dtype = rio.int16
                    # burned = rioft.rasterize(shapes=geom, fill=0)
                    out_class = os.path.join(dirfuncs.guess_data_dir(), 'supplementary_class', landcover, name + '.tif')
                    with rio.open(out_class, 'w+', driver='GTiff',
                                  height=height, width=width,
                                  crs=crs, dtype=dtype, transform=trans, count=1) as out:
                        out_arr = out.read(1)
                        burned = rioft.rasterize(shapes=[(geom[0], supplemental_class_codes[landcover])], fill=-9999,
                                                 out=out_arr, transform=out.transform)
                        burned = np.where(burned != supplemental_class_codes[landcover], -9999,
                                          burned)  # NoData the other pixels
                        out.write_band(1, burned)
                    out.close()
                    get_reference=False

                out_band = os.path.join(dirfuncs.guess_data_dir(), 'supplementary_class', landcover, 'out',
                                        name + '_' + band + '.tif')
                if os.path.isfile(out_band): continue # this is a hack because I have hit errors in the middle of long batch jobs and had find way to restart

                image = image_cache.get_band_by_context_year(band, island, my_year)
                out_img = image.rio.clip(coords, image.rio.crs)
                if out_img.dtype == 'float64':
                    out_img.data = np.float32(out_img)
                dtype = rio.float32


                print('Writing:  ', out_band)
                with rio.open(out_band, 'w+', driver='GTiff',
                              height=height, width=width,
                              crs=crs, dtype=dtype, transform=trans, count=1) as out2:
                    out2.write_band(1, out_img[0])
                out2.close()




def get_bounding_box_polygon(path_to_shape, out_crs='EPSG:4326'):
    print('PATH:  ', path_to_shape)
    ds = ogr.Open(path_to_shape)
    inLayer = ds.GetLayer()
    crs_init = {'init':out_crs}
    print('CRS_INIT: ',crs_init)
    crs = inLayer.GetSpatialRef()
    print('crs:  ', crs)
    in_crs_code = crs.GetAuthorityCode(None)
    print('crs_code:  ',in_crs_code)
    print(type(in_crs_code))
    xmin, xmax, ymin, ymax = [99999999,-99999999,99999999,-99999999]
    for feature in inLayer:
        geom = feature.GetGeometryRef()
        i_xmin, i_xmax, i_ymin, i_ymax = geom.GetEnvelope()
        if(i_xmin<xmin):
            xmin=i_xmin
        if (i_ymin < ymin):
            ymin = i_ymin
        if (i_xmax > xmax):
            xmax = i_xmax
        if (i_ymax > ymax):
            ymax = i_ymax
    print(xmin, ymin, xmax, ymax)
    print(type(xmin), type(ymin), type(xmax), type(ymax))
    bbox = box(xmin, ymin, xmax, ymax)
    try:
        geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=from_epsg(in_crs_code))
    except:
        geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=from_epsg(4326))
    geo = geo.to_crs(crs=out_crs)
    coords = getFeatures(geo)
    print('Coords:  ',coords)
    return(coords)


if __name__ == "__main__":
    nada = np.nan
    print(os.getenv('PROJ_LIB'))
    #get_bounding_box_polygon(input)
    x = ingest_kml_fixed_classes()
    print(x)