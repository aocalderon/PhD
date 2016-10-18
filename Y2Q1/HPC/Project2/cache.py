import collections

CACHE_SIZE = 60
LINE_SIZE = 10
cache = collections.deque([None] * CACHE_SIZE, maxlen = CACHE_SIZE)

n = 100
a = []
for i in range(0,n*n):
	a.append(1)
b = []
for i in range(0,n*n):
	b.append(1)
c = []
for i in range(0,n*n):
	c.append(0)

DA = {}
for i in range(0,n*n):
	DA[i] = 0
DB = {}
for i in range(0,n*n):
	DB[i] = 0
DC = {}
for i in range(0,n*n):
	DC[i] = 0

# i
for i in range(0,n):
	# j
	for j in range(0, n):
		r = c[i*n+j]
		key = "C{0}".format((i*n+j) - (i*n+j) % LINE_SIZE)
		if key not in cache:
			cache.append(key)
			DC[i*n+j] += 1
		else:
			cache.remove(key)
			cache.append(key)
		# k
		for k in range(0,n):
			r = r + (a[i*n+k] * b[k*n+j])
			key = "A{0}".format((i*n+k) - (i*n+k) % LINE_SIZE)
			if key not in cache:
				cache.append(key)
				DA[i*n+k] += 1
			else:
				cache.remove(key)
				cache.append(key)
			key = "B{0}".format((k*n+j) - (k*n+j) % LINE_SIZE)
			if key not in cache:
				cache.append(key)
				DB[k*n+j] += 1
			else:
				cache.remove(key)
				cache.append(key)
			#print cache
		c[i*n+j] = r
		
size = 13
for i in range(0,size):
	for j in range(0,size):
		print "A({0},{1})={2}\t".format(i,j,DA[i*n+j]),
	print "\n"
print ""
for i in range(0,size):
	for j in range(0,size):
		print "B({0},{1})={2}\t".format(i,j,DB[i*n+j]),
	print "\n"
print ""
for i in range(0,size):
	for j in range(0,size):
		print "C({0},{1})={2}\t".format(i,j,DC[i*n+j]),
	print "\n"
print cache
