from osgeo import ogr


def get_bounding_box():
    ds = ogr.Open('ne_10m_admin_0_countries.shp')
    env = geom.GetEnvelope()
    print(env)
    return env