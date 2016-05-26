import xml.dom.minidom
import socket
from geoip import geolite2
import subprocess

doc = xml.dom.minidom.parse("usa.xml")
list = doc.getElementsByTagName("a")
for item in list:
	url = item.getAttribute("href")[7:-1]
	try:
		addr = socket.gethostbyname(url)
		match = geolite2.lookup(addr)
	except socket.error, msg:
		addr
	else:
		try:
			output = subprocess.check_output(["ping","-c1",url])
		except subprocess.CalledProcessError,e:
			url
		else:
			print "{0}".format(url)
			# print "{0};{1};{2};{3}".format(url,addr,match.location[0],match.location[1])
	
