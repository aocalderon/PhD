import socket
from geoip import geolite2
import subprocess
import geocoder
import time

f = open('top10.tsv', 'r')
for url in f.readlines():
	url = url[:-1]
	fields = url.split('\t')
	g = geocoder.google([fields[1],fields[0]], method='reverse')
	time.sleep(2)
	print(g.city+" & "+g.state+" & "+fields[2])
