import networkx as nx
import pandas as pd

G=nx.Graph()
center_y = 29.49
center_x = -98.4
center = (center_x,center_y)
G.add_node(center)

candidates = pd.read_csv('candidates.csv', sep=';', header=None, names=['url','ip','lat','lon','city','state'])
pings = pd.read_csv('pings.csv', sep=';', header=None, names=['url','ping'])

nodes = candidates.loc[:,['url','lat','lon']].merge(pings).loc[:,['lat','lon','ping']]

for index, node in nodes.iterrows():
	point = (node['lon'],node['lat'])
	G.add_node(point)
	G.add_edge(center,point,weight=node['ping'])
nx.write_shp(G,'shapefiles/')


