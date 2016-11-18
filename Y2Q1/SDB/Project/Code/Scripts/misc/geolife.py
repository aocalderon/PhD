#!/usr/bin/python

from os import listdir
from os.path import isfile, join
from pyproj import Proj, transform

counter = 0
output = open("beijing2.csv", "w+")
for directory in listdir("Data/"):
	path = "Data/{0}/Trajectory/".format(directory)
	files = [f for f in listdir(path) if isfile(join(path, f))]
	for file_name in files:
		print("Reading {0}{1} ...".format(path,file_name))
		with open("{0}/{1}".format(path,file_name)) as file:
			for line in file:
				fields = line.split(",")
				if len(fields) > 1 and len(fields) < 8:
					output.write("{5},{0},{1},{2},{3} {4}".format(fields[0],fields[1],fields[4],fields[5],fields[6],counter))
					counter = counter + 1
