library(animation)
library(ggplot2)

ani.options(interval=0.25,
            movie.name = "movie", 
            ani.height = 750, 
            ani.width = 750)
data <- read.csv("Beijing_PBFE2vsPBFE3vsPBFE4_N10K-100K_E10-200_OnCluster.csv", header = F)
ylimit = max(data$V5)
saveGIF({
  for(i in seq(10,100,10)){
    dataset <- paste0(i,"K")
    sample <- data[data$V3 == dataset,c(1,2,5)]
    names(sample) = c("Algorithm","Epsilon", "Time")
    g = ggplot(data=sample, aes(x=factor(Epsilon), y=Time, group=Algorithm, colour=Algorithm, shape=Algorithm)) +
      geom_line(aes(linetype=Algorithm)) +
      geom_point(size=3) +
      labs(title=paste("Execution time in Beijing dataset (N=",dataset,").")) + 
      scale_y_continuous("Time (sec)", limits=c(0,ylimit)) +
      scale_x_discrete(expression(paste(epsilon," (mts)"))) 
    plot(g)
  }  
})