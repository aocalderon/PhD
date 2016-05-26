import subprocess

f = open('dns3.txt', 'r')
for url in f.readlines():
	try:
		url = url[:-1]
		output = subprocess.check_output(["ping","-c5",url])
	except subprocess.CalledProcessError,e:
		url
	else:
		ping = float(output.split('\n')[-2:-1][0].split('/')[-3:-2][0])
		print "{0};{1}".format(url, ping)

