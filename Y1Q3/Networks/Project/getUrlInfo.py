import socket
from geoip import geolite2
import subprocess
import geocoder
import time

f = open('dns.txt', 'r')
for url in f.readlines():
	try:
		url = url[:-1]
		addr = socket.gethostbyname(url)
		match = geolite2.lookup(addr)
		lat = round(match.location[0],2)
		lon = round(match.location[1],2)
		g = geocoder.google([lat,lon], method='reverse')
		time.sleep(2)
	except socket.error, msg:
		print ""
	else:
		print "{0};{1};{2};{3};{4};{5}".format(url,addr,lat,lon,g.city,g.state)
