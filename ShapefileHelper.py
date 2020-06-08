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
from rasterio.mask import mask
from shapely.geometry import box
import glob
import json as js
from shapely.geometry import shape, GeometryCollection
from fiona.crs import from_epsg
from rasterio import features as rioft
import imagery_data

image_cache = imagery_data.Imagery_Cache.getInstance()

#out_tif = r'C:\\Users\\ME\\Dropbox\\HCSproject\\data\\PoC\\Sumatra\\out\\2015\\jambi_crop_blue_max.tif'
input='C:\\Users\\ME\\Dropbox\\HCSproject\\data\\app_files\\stratified_shapefiles\\Jambi_WKS_Stratification.shp'

def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    return [js.loads(gdf.to_json())['features'][0]['geometry']]


def ingest_kml_fixed_classes():
    input = 'C:\\Users\\ME\\Dropbox\\HCSproject\\data\\PoC\\supplementary_class\\impervious\\doc.kml'
    srcDS = gdal.OpenEx(input)
    output = 'C:\\Users\\ME\\Dropbox\\HCSproject\\data\\PoC\\supplementary_class\\impervious\\impervious.json'
    #ds = gdal.VectorTranslate(output, srcDS, format='GeoJSON')
    #file = glob.glob(output)
    with open(output) as f:
        features = js.load(f)["features"]

    # NOTE: buffer(0) is a trick for fixing scenarios where polygons have overlapping coordinates
    #temp = GeometryCollection([shape(feature["geometry"]).buffer(0) for feature in features])
    shapes = ( (shape(feature["geometry"]).buffer(0),int(feature['properties']['Description']), feature['properties']['Name']) for feature in features )

    image2 = image_cache.get_band_by_island_year('nir_max', 'Sumatra', 2015)
    image = image_cache.get_band_by_island_year('nir_max', 'Kalimantan', 2015)
    for geom in shapes:
        print(geom)
        xmin, ymin, xmax, ymax = geom[0].bounds
        name = geom[2]
        bbox = box(xmin, ymin, xmax, ymax)
        geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=from_epsg(4326))
        geo = geo.to_crs(crs=from_epsg(4326))
        coords = getFeatures(geo)
        try:
            out_img = image.rio.clip(coords, image.rio.crs)
        except:
            #try:
                print("CLIP ERROR trying other island")
                out_img = image2.rio.clip(coords, image.rio.crs)
        # meta = out_img.meta.copy()
        trans = out_img.transform
        crs = out_img.rio.crs
        height = out_img.rio.height
        width = out_img.rio.width
        dtype = rio.int16
        # burned = rioft.rasterize(shapes=geom, fill=0)
        out_fn = 'C:\\Users\\ME\\Dropbox\\HCSproject\\data\\PoC\\supplementary_class\\impervious\\' + name +'.tif'
        with rio.open(out_fn, 'w+', driver='GTiff',
                      height=height, width=width,
                      crs=crs, dtype=dtype, transform=trans, count=1) as out:
            out_arr = out.read(1)
            burned = rioft.rasterize(shapes=[(geom[0], geom[1])], fill=-9999, out=out_arr, transform=out.transform)
            burned = np.where(burned != geom[1], -9999, burned) #NoData the other pixels
            out.write_band(1, burned)
        out.close()






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


def read_raster():
    fp = r'C:\\Users\\ME\\Dropbox\\HCSproject\\data\\PoC\\Sumatra\\out\\2015\\blue_max.tif'
    data = rio.open(fp)
    print('Raster CRS:  ', data.crs.data)
  ##  out_img, out_transform = mask(raster=data, shapes=coords, crop=True)
  #  epsg_code = out_crs
  #   srs = osr.SpatialReference()  ###
  #   srs.SetFromUserInput(epsg_code)  ###
  #   proj4_str = srs.ExportToProj4()
  #   print(proj4_str)
  #   out_meta = data.meta.copy()
  #   out_meta.update({"driver": "GTiff",
  #                    "height": out_img.shape[1],
  #                    "width": out_img.shape[2],
  #                    "transform": out_transform,
  #                    "crs": proj4_str}
  #                   )
  #   with rasterio.open(out_tif, "w", **out_meta) as dest:
  #       dest.write(out_img)
if __name__ == "__main__":
    nada = np.nan
    print(os.getenv('PROJ_LIB'))
    #get_bounding_box_polygon(input)
    x = ingest_kml_fixed_classes()
    print(x)
    #read_raster()
    # bobox=box(235186.7964500059, 9825476.572053133 ,379883.6319000004 ,9916487.528701443)
    # geo = gpd.GeoDataFrame({'geometry': bobox}, index=[0], crs=from_epsg('32748'))
    # print(geo)
    # crs_init = 'EPSG:4326'
    # geo = geo.to_crs(crs=crs_init)
    # coords = getFeatures(geo)
    # print('Coords:  ', coords)