
###
# Execution time and performance for dgemm0, dgemm1 and dgemm2
###

data <- "64 7.128216 4.931031 1.803537
128 39.969206 25.759388 8.4104
256 325.499361 166.953458 54.454552
512 3049.546593 2179.83658 783.193455
1024 31205.88699 23135.24844 8132.992738
2048 420966.9449 285941.5489 151813.1982"
data <- read.csv(textConnection(data), sep = " ", header = F)

data$GFLOPS1 = (((2 * data$V1^3) / data$V2) * 1000) / 1000000000
data$GFLOPS2 = (((2 * data$V1^3) / data$V3) * 1000) / 1000000000
data$GFLOPS3 = (((2 * data$V1^3) / data$V4) * 1000) / 1000000000

print("Execution time and performance for dgemm0, dgemm1 and dgemm2...")
print(data)

###
# Execution time and performance for dgemm0, dgemm1, dgemm2 and dgemm3
###

data <- "66 6.621722 4.816092 1.693514 0.759508
132 34.817608 25.360189 8.634621 6.476681
258 321.320101 198.58523 67.098006 45.724277
516 2348.836613 1585.794235 612.967972 435.972731
1026 18372.79847 13365.86185 4980.412193 2743.350511
2052 219459.9502 123244.001 101395.2948 39827.554831"
data <- read.csv(textConnection(data), sep = " ", header = F)

data$GFLOPS1 = (((2 * data$V1^3) / data$V2) * 1000) / 1000000000
data$GFLOPS2 = (((2 * data$V1^3) / data$V3) * 1000) / 1000000000
data$GFLOPS3 = (((2 * data$V1^3) / data$V4) * 1000) / 1000000000
data$GFLOPS4 = (((2 * data$V1^3) / data$V5) * 1000) / 1000000000

print("Execution time and performance for dgemm0, dgemm1, dgemm2 and dgemm3...")
print(data)

###
# Cache miss rate for 10000x10000 matrix
###

n = 10000
b = 10
data <- ""
R=(n^2*(1/b) + n^2*n*(1/b) + n^2*n) / (2*n^3 + n^2)
data = paste0(data, paste0("IJK; cij=1 if j%b=0; aik=n if k%b=0; bkj=n; ~50%+50%/", b, " ",round(R*100,3),"%"), "\n")
R=(n^2 + n^2*n*(1/b) +	n^2*n) / (2*n^3 + n^2)	
data = paste0(data, paste0("JIK; cij=1; aik=n if k%b=0; bkj=n; ~50%+50%/", b, " ",round(R*100,3),"%"), "\n")
R=(n^2*n*(1/b) + n^2 + n^2*n*(1/b)) / (2*n^3 + n^2)
data = paste0(data, paste0("IKJ; cij=n if j%b=0; aik=1; bkj=n if j%b=0; ~100%/", b, " ",round(R*100,3),"%"), "\n")
R=(n^2*n*(1/b) + n^2 + n^2*n*(1/b)) / (2*n^3 + n^2)
data = paste0(data, paste0("KIJ; cij=n if j%b=0; aik=1; bkj=n if j%b=0; ~100%/", b, " ",round(R*100,3),"%"), "\n")
R=(n^2*n + n^2*n + n^2) / (2*n^3 + n^2)
data = paste0(data, paste0("KJI; cij=n; aik=n; bkj=1; ~100% ",round(R*100,3),"%"), "\n")
R=(n^2*n + n^2*n + n^2) / (2*n^3 + n^2)
data = paste0(data, paste0("JKI; cij=n; aik=n; bkj=1; ~100% ",round(R*100,3),"%"), "\n")
data <- read.csv(textConnection(data), sep = ";", header = F)
print(paste0("Cache miss rate for 10000x10000 matrix: n=", n, " b=", b))
print(data)

###
# Cache miss rate for 10x10 matrix
###

n = 10
b = 10
data <- ""
R=(n^2*(1/b) + n^2*(1/b) + n^2*(1/b)) / (2*n^3 + n^2)
data = paste0(data, paste0("cij=1 if j%b=0; aik=1 if k%b=0; bkj=1 if j%b=0; ~150%/", b*n, " ",round(R*100,3),"%"), "\n")
data <- read.csv(textConnection(data), sep = ";", header = F)
print(paste0("Cache miss rate for 10x10 matrix: n=", n, " b=", b))
print(data)

###
# Cache miss rate for 10000x10000 matrix using blocking
###

n = 10000
b = 10
B = 10
data <- ""
R = (n^2*(1/b) + n^2*(n/B)*(1/b) + n^2*(n/B)*(1/b)) / (2*n^3+n^2)
data = paste0(data, paste0("IJK&JIK; cij=1 if j%b=0; aik=n/B if k%b=0; bkj=n/B if j%b=0; ~100%/", b*B, " ",round(R*100,3),"%"), "\n")
R = (n^2*(n/B)*(1/b) + n^2*(1/b) + n^2*(n/B)*(1/b)) / (2*n^3+n^2)
data = paste0(data, paste0("IKJ&KIJ; cij=n/B if j%b=0; aik=1 if k%b=0; bkj=n/B if j%b=0; ~100%/", b*B, " ",round(R*100,3),"%"), "\n")
R = (n^2*(n/B)*(1/b) + n^2*(n/B)*(1/b) + n^2*(1/b)) / (2*n^3+n^2)
data = paste0(data, paste0("JKI&KJI; cij=n/B if j%b=0; aik=b/B if k%b=0; bkj=1 if j%b=0; ~100%/", b*B, " ",round(R*100,3),"%"), "\n")
data <- read.csv(textConnection(data), sep = ";", header = F)
print(paste0("Cache miss rate for 10000x10000 matrix using blocking: n=",n," b=",b," B=",B))
print(data)