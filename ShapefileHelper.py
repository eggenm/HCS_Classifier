from osgeo import ogr, osr
import rasterio
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd
from fiona.crs import from_epsg
import json

out_tif = r'C:\\Users\\ME\\Dropbox\\HCSproject\\data\\PoC\\Sumatra\\out\\2015\\jambi_crop_blue_max.tif'

def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

def get_bounding_box():
    ds = ogr.Open('C:\\Users\\ME\\Dropbox\\HCSproject\\data\\app_files\\stratified_shapefiles\\Jambi_WKS_Stratification.shp')
    inLayer = ds.GetLayer()
    crs = inLayer.GetSpatialRef()
    print(crs)
    i=0
    xmin, xmax, ymin, ymax = [99999999999,-99999999999999,99999999999,-9999999999999]
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
    bbox = box(xmin, ymin, xmax, ymax)
    geo = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=from_epsg(32748))
    fp = r'C:\\Users\\ME\\Dropbox\\HCSproject\\data\\PoC\\Sumatra\\out\\2015\\blue_max.tif'
    data = rasterio.open(fp)
    print('Raster CRS:  ',data.crs.data)
    geo = geo.to_crs(crs=data.crs.data)
    coords = getFeatures(geo)
    print('Coords:  ',coords)
    out_img, out_transform = mask(raster=data, shapes=coords, crop=True)
    epsg_code = data.crs.data['init']
    srs = osr.SpatialReference()  ###
    srs.SetFromUserInput(epsg_code)  ###
    proj4_str = srs.ExportToProj4()
    print(proj4_str)
    out_meta = data.meta.copy()
    out_meta.update({"driver": "GTiff",
        "height": out_img.shape[1],
        "width": out_img.shape[2],
        "transform": out_transform,
        "crs": proj4_str}
    )
    with rasterio.open(out_tif, "w", **out_meta) as dest:
        dest.write(out_img)

def read_raster():
    print('')

get_bounding_box()