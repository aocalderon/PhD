import sys
import math

def f1(x):
	return math.pow(x,3)-3*math.pow(x,2)+3*x-1
def f1_prime(x):
	return 3*math.pow(x,2)-6*x+3

tol = 0.000001

## Bisect
print "\nBisect\n"
a = 0
b = 2
print "{0:.6f}\t{1:.6f}\t{2:.6f}\t{3:.6f}".format(a,f1(a),b,f1(b)) 
while b - a > tol:
	m = a + (b - a)/2.0
	if math.copysign(1,f1(a)) == math.copysign(1,f1(m)):
		a = m
	else:
		b = m
	print "{0:.6f}\t{1:.6f}\t{2:.6f}\t{3:.6f}".format(a,f1(a),b,f1(b)) 

## Newton's
print "\nNewton's\n"
x = 2
print "{0:.6f}\t{1:.6f}\t{2:.6f}".format(x,f1(x),f1_prime(x)) 
while math.fabs(f1(x)) > tol:
	x = x - f1(x)/f1_prime(x)
	print "{0:.6f}\t{1:.6f}\t{2:.6f}".format(x,f1(x),f1_prime(x)) 

## Secant
print "\nSecant\n"
x0 = 0
x1 = 3
print "{0:.6f}\t{1:.6f}".format(x0,f1(x0)) 
while math.fabs(f1(x0)) > tol:
	x2 = x1 - f1(x1)*((x1-x0)/(f1(x1)-f1(x0)))
	x0 = x1
	x1 = x2
	print "{0:.6f}\t{1:.6f}".format(x0,f1(x0)) 
