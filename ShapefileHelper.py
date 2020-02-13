from osgeo import ogr, osr
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd
from fiona.crs import from_epsg
import json

#out_tif = r'C:\\Users\\ME\\Dropbox\\HCSproject\\data\\PoC\\Sumatra\\out\\2015\\jambi_crop_blue_max.tif'
#input='C:\\Users\\ME\\Dropbox\\HCSproject\\data\\app_files\\stratified_shapefiles\\Jambi_WKS_Stratification.shp'

def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

def get_bounding_box_polygon(path_to_shape, out_crs='EPSG:4326'):
    print('PATH:  ', path_to_shape)
    ds = ogr.Open(path_to_shape)
    inLayer = ds.GetLayer()
    crs_init = {'init':out_crs}
    print('CRS_INIT: ',crs_init)
    crs = inLayer.GetSpatialRef()
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
    geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=from_epsg(int(in_crs_code)))
    geo = geo.to_crs(crs=crs_init)
    coords = getFeatures(geo)
    print('Coords:  ',coords)
    return(coords)


def read_raster():
    fp = r'C:\\Users\\ME\\Dropbox\\HCSproject\\data\\PoC\\Sumatra\\out\\2015\\blue_max.tif'
    data = rasterio.open(fp)
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
    #get_bounding_box_polygon(input)
    #read_raster()
    bobox=box(235186.7964500059, 9825476.572053133 ,379883.6319000004 ,9916487.528701443)
    geo = gpd.GeoDataFrame({'geometry': bobox}, index=[0], crs=from_epsg('32748'))
    #print(geo)
    crs_init = {'init': 'EPSG:4326'}
    geo = geo.to_crs(crs=crs_init)
    coords = getFeatures(geo)
    print('Coords:  ', coords)