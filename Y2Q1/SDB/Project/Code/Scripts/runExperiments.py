#!/usr/bin/env python

import sys
import psutil 
import subprocess


command = ['/home/and/Documents/PhD/Code/Y2Q1/SDB/Project/Code/Scripts/bfe/bfe.py','100','1','2','/opt/Datasets/Beijing/P100K.csv']
proc = subprocess.Popen(command)
pid = proc.pid
process = psutil.Process(pid)

# Poll process for new output until finished
while True:
	if proc.poll() is None:
		print("{0}\t{1}\n".format(process.memory_percent(), process.cpu_percent()))
