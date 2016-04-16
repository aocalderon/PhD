start <- 219
top <- 10
xlab <- "Cache block size"
labels <- c('32','64','128','256')

data <- readLines("Results/results_CacheBS32.txt") 
data <- data[start:length(data) - 1]
n <- length(data)
empty <- rep(0, n)
results <- data.frame(metric = empty, C1 = empty)
for(i in 1:n){
  result <- strsplit(strsplit(data[i], "#")[[1]][1], "\\s+")[[1]]
  results[i,1] <- result[1]
  results[i,2] <- as.numeric(result[2])
}
data <- readLines("Results/results_CacheBS64.txt") 
data <- data[start:length(data) - 1]
results$C2 <- empty
for(i in 1:n){
  result <- strsplit(strsplit(data[i], "#")[[1]][1], "\\s+")[[1]]
  results[i,3] <- as.numeric(result[2])
}
data <- readLines("Results/results_CacheBS128.txt") 
data <- data[start:length(data) - 1]
results$C4 <- empty
for(i in 1:n){
  result <- strsplit(strsplit(data[i], "#")[[1]][1], "\\s+")[[1]]
  results[i,4] <- as.numeric(result[2])
}
data <- readLines("Results/results_CacheBS256.txt") 
data <- data[start:length(data) - 1]
results$C8 <- empty
results$desc <- empty
for(i in 1:n){
  result <- strsplit(strsplit(data[i], "#")[[1]][1], "\\s+")[[1]]
  results[i,5] <- as.numeric(result[2])
  results[i,6] <- strsplit(data[i], "#")[[1]][2]
}

plot(1:4,results[15,2:5],type='l',axes=F,ylab=results[15,1]
     ,main=results[15,6]
     ,xlab=xlab)
points(1:4,results[15,2:5],cex=0.7,pch=21,bg=1)
box()
axis(1,at=1:4,labels=labels)
axis(2)

results$min <- apply(results[,2:5],1,min)
results$max <- apply(results[,2:5],1,max)
results$range <- results$max - results$min
results$sd <- apply(results[,2:5],1,sd)
results$index <- results$sd / results$range

results <- results[with(results,order(-index)),]
write.csv(results,'Results/metrics_CacheBS.csv',row.names = F)
results <- results[1:top,1:6]

for(i in 1:top){
  plot(1:4,results[i,2:5],type='l',axes=F,ylab=results[i,1]
       ,main=results[i,6]
       ,xlab=xlab)
  box()
  axis(1,at=1:4,labels=labels)
  axis(2)
  points(1:4,results[i,2:5],cex=0.7,pch=21,bg=1)
}