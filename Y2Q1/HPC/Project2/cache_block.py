import collections

CACHE_SIZE = 60
LINE_SIZE = 10
BLOCK_SIZE = 10
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
for i in range(0,n,BLOCK_SIZE):
	# j
	for j in range(0, n,BLOCK_SIZE):
		# k
		for k in range(0,n,BLOCK_SIZE):
			for ii in range(i,i+BLOCK_SIZE):
				for jj in range(j,j+BLOCK_SIZE ):
					r = c[ii*n+jj]
					key = "C{0}".format((ii*n+jj) - (ii*n+jj) % LINE_SIZE)
					if key not in cache:
						cache.append(key)
						DC[ii*n+jj] += 1
					else:
						cache.remove(key)
						cache.append(key)					
					for kk in range(k,k+BLOCK_SIZE):
						r = r + (a[ii*n+kk] * b[kk*n+jj])
						key = "A{0}".format((ii*n+kk) - (ii*n+kk) % LINE_SIZE)
						if key not in cache:
							cache.append(key)
							DA[ii*n+kk] += 1
						else:
							cache.remove(key)
							cache.append(key)
						key = "B{0}".format((kk*n+jj) - (kk*n+jj) % LINE_SIZE)
						if key not in cache:
							cache.append(key)
							DB[kk*n+jj] += 1
						else:
							cache.remove(key)
							cache.append(key)
						#print cache
					c[ii*n+jj] = r
		
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
