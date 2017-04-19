#!/opt/miniconda2/bin/python

import pandas as pd
import folium
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())

epsilon = 100
the_zoom = 17

points = pd.read_csv('points.csv')
disks = pd.read_csv('disks.csv')
links = pd.read_csv('links.csv')

sql = """
    SELECT 
        l.pid AS pid, l.dids AS dids, p.lat AS lat, p.lng AS lng 
    FROM 
        (SELECT 
            pid, GROUP_CONCAT(did, ' ') AS dids 
        FROM 
            links 
        GROUP BY 
            pid) AS l
    JOIN
        points AS p
    USING
        (pid);
"""
p = pysqldf(sql)

sql = """
    SELECT 
        l.did AS did, l.pids AS pids, d.lat AS lat, d.lng AS lng 
    FROM 
        (SELECT 
            did, GROUP_CONCAT(pid, ' ') AS pids 
        FROM 
            links 
        GROUP BY 
            did) AS l
    JOIN
        disks AS d
    USING
        (did);
"""
d = pysqldf(sql)

the_map = folium.Map(location=[39.97537, 116.33127], zoom_start=the_zoom)

the_disks = folium.FeatureGroup(name="Disks")
d.apply(lambda row:folium.CircleMarker(
    location=[row["lat"], row["lng"]],
    popup="DID:{0} => [{1}]".format(row["did"],row["pids"]),
    color='red', 
    fill_color='red', 
    fill_opacity=0.05, 
    radius=epsilon/2
).add_to(the_disks), axis=1)

the_points = folium.FeatureGroup(name="Points")
p.apply(lambda row:folium.RegularPolygonMarker(
    location=[row["lat"], row["lng"]],
    popup="PID:{0} => [{1}]".format(row["pid"],row["dids"]),
    color='purple',
    radius=3
).add_to(the_points), axis=1)

the_map.add_child(the_points)
the_map.add_child(the_disks)
folium.LayerControl().add_to(the_map)
the_map.save('prune.html')

p.to_csv('tpoints.dat', header=False, index=False, columns=['dids'], doublequote=False)
d.to_csv('tdisks.dat', header=False, index=False, columns=['pids'], doublequote=False)

p.to_csv('p_points.dat', header=False, index=False, columns=['pid', 'dids'], doublequote=False)
d.to_csv('p_disks.dat', header=False, index=False, columns=['did', 'pids'], doublequote=False)
