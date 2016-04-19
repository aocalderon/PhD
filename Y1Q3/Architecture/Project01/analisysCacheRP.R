top <- 10
labels = c("FIFO", "LRU", "Random")
app_prefix <- '_G'
fea_prefix <- '_CacheRP'

xlab <- "Cache replacement policy"

data <- readLines(paste0("Results/results",app_prefix,fea_prefix,"F.txt")) 
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
data <- readLines(paste0("Results/results",app_prefix,fea_prefix,"L.txt")) 
data <- data[start:length(data) - 1]
results$C2 <- empty
for(i in 1:n){
  result <- strsplit(strsplit(data[i], "#")[[1]][1], "\\s+")[[1]]
  results[i,3] <- as.numeric(result[2])
}
data <- readLines(paste0("Results/results",app_prefix,fea_prefix,"R.txt")) 
data <- data[start:length(data) - 1]
results$C4 <- empty
results$desc <- empty
for(i in 1:n){
  result <- strsplit(strsplit(data[i], "#")[[1]][1], "\\s+")[[1]]
  results[i,4] <- as.numeric(result[2])
  results[i,5] <- strsplit(data[i], "#")[[1]][2]
}

pos_cpi <- match("sim_CPI",results$metric)
plot(1:3,results[pos_cpi,2:4],type='l',axes=F,ylab=results[pos_cpi,1]
     ,main=results[pos_cpi,5]
     ,xlab=xlab)
points(1:3,results[pos_cpi,2:4],cex=0.7,pch=21,bg=1)
box()
axis(1,at=1:3,labels=labels)
axis(2)

results = results[complete.cases(results),]
results$min <- apply(results[,2:4],1,min)
results$max <- apply(results[,2:4],1,max)
results$range <- results$max - results$min
results$sd <- apply(results[,2:4],1,sd)
results$index <- results$sd / results$range

results <- results[with(results,order(-index)),]
write.csv(results,paste0("Results/metrics",app_prefix,fea_prefix,".csv"),row.names = F)
results <- results[1:top,1:5]

for(i in 1:top){
  plot(1:3,results[i,2:4],type='l',axes=F,ylab=results[i,1]
       ,main=results[i,5]
       ,xlab=xlab)
  box()
  axis(1,at=1:3,labels=labels)
  axis(2)
  points(1:3,results[i,2:4],cex=0.7,pch=21,bg=1)
}