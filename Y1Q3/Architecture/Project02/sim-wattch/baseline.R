path = "/home/and/Downloads/"
command = paste0(path,"sim-wattch/sim-outorder ",path,"benchmarks/go.alpha 2&> baseline.txt")
system(command)
data <- readLines("baseline.txt")
start <- match("sim: ** simulation statistics **", data)
end <- match("mem.ptab_miss_rate           0.0004 # first level page table miss rate", data)
data <- data[start:end]
n <- length(data)
empty <- rep(0, n)
results <- data.frame(metric = empty, C1 = empty)
for(i in 1:n){
  result <- strsplit(strsplit(data[i], "#")[[1]][1], "\\s+")[[1]]
  results[i,1] <- result[1]
  results[i,2] <- as.numeric(result[2])
}
results <- results[complete.cases(results),]
