#!/opt/miniconda3/bin/python

import numpy as np
import pandas as pd
import folium
from pyproj import Proj

def makeWKT(points):
    wkt = "POLYGON (({0} {1}, {2} {1}, {2} {3}, {0} {3}, {0} {1}))".format(points['Lon1'], points['Lat1'], points['Lon2'], points['Lat2'])
    return pd.Series([points['ID'], wkt])

proj4769 = Proj("+proj=tmerc +lat_0=0 +lon_0=126 +k=1 +x_0=500000 +y_0=0 +ellps=krass +units=m +no_defs")
mbrs = pd.read_csv('B1K_RTree_MBRs.csv', header=None)
mbrs.columns = ['id', 'x1', 'y1', 'x2', 'y2']

lon1, lat1 = proj4769(mbrs['x1'].values, mbrs['y1'].values, inverse=True)
lon2, lat2 = proj4769(mbrs['x2'].values, mbrs['y2'].values, inverse=True)

df = pd.DataFrame(np.c_[mbrs['id'].values, lat1, lon1, lat2, lon2], columns=['ID', 'Lat1', 'Lon1', 'Lat2', 'Lon2'])
df.apply(makeWKT, axis=1).to_csv('test.wkt', header=None, index=False)

"""
the_map = folium.Map(location=[39.93644, 116.38108], zoom_start=13)
the_mbrs = folium.FeatureGroup(name="MBRs")
mbrs.apply(lambda mbr:folium.features.RectangleMarker(
    bounds=[[mbr[2], mbr[1]], [mbr[4], mbr[3]]],
    popup="MBR ID:{0} => [[{1},{2}];[{3},{4}]]".format(mbr[0],mbr[2], mbr[1], mbr[4], mbr[3]),
    color='blue', 
    fill_color='blue'
).add_to(the_mbrs), axis=1)
the_map.add_child(the_mbrs)
folium.LayerControl().add_to(the_map)
the_map.save('MBRs.html')
"""