import networkx as nx

G=nx.Graph()
center_y = 29.49
center_x = -98.4
G.add_node((center_x,center_y))

f = open('url_locations.csv', 'r')
for record in f.readlines():
	fields = record.split(';')
	G.add_node((fields[3],fields[2]))
	G.add_edge((center_x,center_y),(fields[3],fields[2]),weight=0.6)
nx.write_shp(G,'/tmp/')


