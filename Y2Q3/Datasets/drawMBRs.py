#!/opt/miniconda2/bin/python

import pandas as pd
import folium
from pyproj import Proj, transform

inProj = Proj(init='epsg:4799')
outProj = Proj(init='epsg:4326')


points = pd.read_csv('B89_RTree.csv', header=None)
points.apply(lambda point: transform(inProj,outProj, point[1], point[2]))
mbrs = pd.read_csv('B89_RTree_MBRs.csv', header=None)

the_map = folium.Map(location=[39.93644, 116.38108], zoom_start=14)

the_points = folium.FeatureGroup(name="Points")
points.apply(lambda point:
    x,y = transform(inProj,outProj, point[1], point[2])
    folium.RegularPolygonMarker(
    location=[x, y],
    popup="MBR ID: {0}".format(point[0]),
    radius=2
).add_to(the_points), axis=1)

#the_mbrs = folium.FeatureGroup(name="MBRs")
#mbrs.apply(lambda mbr:folium.RectangleMarker(
#    bounds=[[mbr[1], mbr[2]], [mbr[3], mbr[4]]],
#    popup="MBR ID:{0}".format(mbr[0]),
#    color='blue', 
#    fill_color='blue', 
#    fill_opacity=0.01
#).add_to(the_mbrs), axis=1)

the_map.add_child(the_points)
#the_map.add_child(the_mbrs)
folium.LayerControl().add_to(the_map)
the_map.save('MBRs.html')
