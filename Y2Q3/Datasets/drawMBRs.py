#!/opt/miniconda3/bin/python

import pandas as pd
import folium
from pyproj import Proj, transform

inProj = Proj(init='epsg:4799')
outProj = Proj(init='epsg:4326')

points = pd.read_csv('B1K_RTree.csv', header=None)

def convertCoords1(point):
    x, y = transform(inProj, outProj, point[1], point[2])
    return pd.Series([x, y])

points[[1, 2]] = points.apply(convertCoords1, axis=1)

mbrs = pd.read_csv('B1K_RTree_MBRs.csv', header=None)

def convertCoords2(points):
    x1, y1 = transform(inProj, outProj, points[1], points[2])
    x2, y2 = transform(inProj, outProj, points[3], points[4])
    return pd.Series([x1, y1, x2, y2])

mbrs[[1, 2, 3, 4]] = mbrs.apply(convertCoords2, axis=1)

the_map = folium.Map(location=[39.93644, 116.38108], zoom_start=13)

the_points = folium.FeatureGroup(name="Points")
points.apply(lambda point:folium.features.CircleMarker(
    location=[point[2], point[1]],
    popup="MBR ID: {0} => [{1},{2}]".format(point[0], point[2], point[1]),
    radius=2
).add_to(the_points), axis=1)

the_mbrs = folium.FeatureGroup(name="MBRs")
mbrs.apply(lambda mbr:folium.features.RectangleMarker(
    bounds=[[mbr[2], mbr[1]], [mbr[4], mbr[3]]],
    popup="MBR ID:{0} => [[{1},{2}];[{3},{4}]]".format(mbr[0],mbr[2], mbr[1], mbr[4], mbr[3]),
    color='blue', 
    fill_color='blue'
).add_to(the_mbrs), axis=1)

the_map.add_child(the_points)
the_map.add_child(the_mbrs)
folium.LayerControl().add_to(the_map)
the_map.save('MBRs.html')
