import networkx as nx
import pandas as pd

G=nx.Graph()
## Riverside
center_y = 33.9680
center_x = -117.3194
center = (center_x,center_y)
G.add_node(center)

candidates = pd.read_csv('candidates.csv', sep=';', header=None, names=['url','ip','lat','lon','city','state'])
pings = pd.read_csv('pings.csv', sep=';', header=None, names=['url','ping'])

nodes = candidates.loc[:,['url','lat','lon']].merge(pings).loc[:,['lat','lon','ping']]

for index, node in nodes.iterrows():
	point = (node['lon'],node['lat'])
	G.add_node(point)
	G.add_edge(center,point,weight=node['ping'],cdn='Riverside')
	
## Canada East 40.86.227.130 Quebec	Quebec
center_y = 46.7933
center_x = -71.2453
center = (center_x,center_y)
G.add_node(center)

pings = pd.read_csv('cdnce.csv', sep=';', header=None, names=['url','ping'])

nodes = candidates.loc[:,['url','lat','lon']].merge(pings).loc[:,['lat','lon','ping']]

for index, node in nodes.iterrows():
	point = (node['lon'],node['lat'])
	G.add_node(point)
	G.add_edge(center,point,weight=node['ping'],cdn='Quebec')
	
## USA East 40.76.196.176 Boydton Virginia
center_y = 36.6648
center_x = -78.3715
center = (center_x,center_y)
G.add_node(center)

pings = pd.read_csv('cdneu.csv', sep=';', header=None, names=['url','ping'])

nodes = candidates.loc[:,['url','lat','lon']].merge(pings).loc[:,['lat','lon','ping']]

for index, node in nodes.iterrows():
	point = (node['lon'],node['lat'])
	G.add_node(point)
	G.add_edge(center,point,weight=node['ping'],cdn='Boydton')
	
## USA South Central 104.210.157.99 San Antonio Texas
center_y = 29.4244
center_x = -98.4842
center = (center_x,center_y)
G.add_node(center)

pings = pd.read_csv('cdnsc.csv', sep=';', header=None, names=['url','ping'])

nodes = candidates.loc[:,['url','lat','lon']].merge(pings).loc[:,['lat','lon','ping']]

for index, node in nodes.iterrows():
	point = (node['lon'],node['lat'])
	G.add_node(point)
	G.add_edge(center,point,weight=node['ping'],cdn='San Antonio')
	
## USA West 13.93.154.132 San Jose California
center_y = 37.3394
center_x = -121.8950
center = (center_x,center_y)
G.add_node(center)

pings = pd.read_csv('cdnwu.csv', sep=';', header=None, names=['url','ping'])

nodes = candidates.loc[:,['url','lat','lon']].merge(pings).loc[:,['lat','lon','ping']]

for index, node in nodes.iterrows():
	point = (node['lon'],node['lat'])
	G.add_node(point)
	G.add_edge(center,point,weight=node['ping'],cdn='San Jose')
	
## Canada Central 13.88.249.118 Toronto Ontario
center_y = 43.6683
center_x = -79.4205
center = (center_x,center_y)
G.add_node(center)

pings = pd.read_csv('cdncc.csv', sep=';', header=None, names=['url','ping'])

nodes = candidates.loc[:,['url','lat','lon']].merge(pings).loc[:,['lat','lon','ping']]

for index, node in nodes.iterrows():
	point = (node['lon'],node['lat'])
	G.add_node(point)
	G.add_edge(center,point,weight=node['ping'],cdn='Toronto')
	
## Central USA 13.67.214.40 Des Moines Iowa
center_y = 41.5921
center_x = -93.6040
center = (center_x,center_y)
G.add_node(center)

pings = pd.read_csv('cdncu.csv', sep=';', header=None, names=['url','ping'])

nodes = candidates.loc[:,['url','lat','lon']].merge(pings).loc[:,['lat','lon','ping']]

for index, node in nodes.iterrows():
	point = (node['lon'],node['lat'])
	G.add_node(point)
	G.add_edge(center,point,weight=node['ping'],cdn='Des Moines')
	
## North Central USA 23.96.185.34 Chicago Illinois
center_y = 41.8826
center_x = -87.6292
center = (center_x,center_y)
G.add_node(center)

pings = pd.read_csv('cdncu.csv', sep=';', header=None, names=['url','ping'])

nodes = candidates.loc[:,['url','lat','lon']].merge(pings).loc[:,['lat','lon','ping']]

for index, node in nodes.iterrows():
	point = (node['lon'],node['lat'])
	G.add_node(point)
	G.add_edge(center,point,weight=node['ping'],cdn='Chicago')

## nx.write_shp(G,'shapefiles/')

cdns = [(-117.3194,33.9680),
(-71.2453, 46.7933),
(-78.3715,36.6648),
(-98.4842,29.4244),
(-121.8950,37.3394),
(-79.4205,43.6683),
(-93.6040,41.5921),
(-87.6292,41.8826)]

c = 0
i = 0
nodes = pd.DataFrame(columns=['n','Node_lat','Node_lon','Average'])
for node in G.nodes_iter():
	flag = True
	for cdn in cdns:
		if cdn == node:
			flag = False
			continue;
	if flag:
		s = 0
		for edge in G.edges_iter():
			if node == edge[0]:
				s = s + G[node][edge[1]]['weight']
		a = s / 8
		if a == 0:
			c = c + 1
		print "{2}. For node {0} the average is {1}".format(node,a,i)
		nodes.loc[i] = [i,node[0],node[1],a]
		i = i + 1
nodes.sort_values(by='Average', ascending=[0]).to_csv('rank.csv')


print "How many 0s: {0}".format(c)

