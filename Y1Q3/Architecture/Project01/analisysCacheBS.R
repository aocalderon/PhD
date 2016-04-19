top <- 10
labels = c("32", "64", "128", "256")
app_prefix <- '_C'
fea_prefix <- '_CacheBS'

xlab <- "Cache block size"

data <- readLines(paste0("Results/results",app_prefix,fea_prefix,"32.txt")) 
start <- match("sim: ** simulation statistics **", data)
data <- data[start:length(data) - 1]
n <- length(data)
empty <- rep(0, n)
results <- data.frame(metric = empty, C1 = empty)
for(i in 1:n){
  result <- strsplit(strsplit(data[i], "#")[[1]][1], "\\s+")[[1]]
  results[i,1] <- result[1]
  results[i,2] <- as.numeric(result[2])
}
data <- readLines(paste0("Results/results",app_prefix,fea_prefix,"64.txt")) 
data <- data[start:length(data) - 1]
results$C2 <- empty
for(i in 1:n){
  result <- strsplit(strsplit(data[i], "#")[[1]][1], "\\s+")[[1]]
  results[i,3] <- as.numeric(result[2])
}
data <- readLines(paste0("Results/results",app_prefix,fea_prefix,"128.txt")) 
data <- data[start:length(data) - 1]
results$C4 <- empty
for(i in 1:n){
  result <- strsplit(strsplit(data[i], "#")[[1]][1], "\\s+")[[1]]
  results[i,4] <- as.numeric(result[2])
}
data <- readLines(paste0("Results/results",app_prefix,fea_prefix,"256.txt")) 
data <- data[start:length(data) - 1]
results$C8 <- empty
results$desc <- empty
for(i in 1:n){
  result <- strsplit(strsplit(data[i], "#")[[1]][1], "\\s+")[[1]]
  results[i,5] <- as.numeric(result[2])
  results[i,6] <- strsplit(data[i], "#")[[1]][2]
}
pos_cpi <- match("sim_CPI",results$metric)
plot(1:4,results[pos_cpi,2:5],type='l',axes=F,ylab=results[pos_cpi,1]
     ,main=results[pos_cpi,6]
     ,xlab=xlab)
points(1:4,results[pos_cpi,2:5],cex=0.7,pch=21,bg=1)
box()
axis(1,at=1:4,labels=labels)
axis(2)

results = results[complete.cases(results),]
results$min <- apply(results[,2:5],1,min)
results$max <- apply(results[,2:5],1,max)
results$range <- results$max - results$min
results$sd <- apply(results[,2:5],1,sd)
results$index <- results$sd / results$range

results <- results[with(results,order(-index)),]
write.csv(results,paste0("Results/metrics",app_prefix,fea_prefix,".csv"),row.names = F)
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