from networkx import *

G=nx.dodecahedral_graph()
nx.draw(G)
nx.draw(G,pos=nx.spring_layout(G))

G = nx.complete_graph(5)
preds = nx.resource_allocation_index(G)
for u, v, p in preds:
	'(%d, %d) -> %.8f' % (u, v, p)
