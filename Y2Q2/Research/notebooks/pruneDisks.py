#!/opt/miniconda2/bin/python

import pandas as pd
import folium

epsilon = 100

data = pd.read_csv('data.csv')
disks = pd.read_csv('disks.csv')
#df1 = disks.iloc[:,[2,3]]
#df1.columns = ['lng', 'lat']
#df2 = disks.iloc[:,[4,5]]
#df2.columns = ['lng', 'lat']
#disks = df1.append(df2, ignore_index=True)
#disks.to_csv('disks2.csv', index=False)

the_map = folium.Map(location=[39.976057, 116.330243], zoom_start=15)
the_disks = folium.FeatureGroup(name="Disks")
disks.apply(lambda row:folium.CircleMarker(location=[row["lat"], row["lng"]],
    color='red', fill_color='red', fill_opacity=0.25, radius=epsilon/2).add_to(the_disks), axis=1)
the_points = folium.FeatureGroup(name="Points")
data.apply(lambda row:folium.RegularPolygonMarker(location=[row["lat"], row["lng"]],
    radius=2).add_to(the_points), axis=1)
the_map.add_child(the_points)
the_map.add_child(the_disks)
folium.LayerControl().add_to(the_map)
the_map.save('prune.html')
