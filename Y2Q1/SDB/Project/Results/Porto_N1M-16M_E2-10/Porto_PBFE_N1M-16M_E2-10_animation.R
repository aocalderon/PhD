library(animation)
library(ggplot2)

ani.options(interval=0.5,
            movie.name = "Porto_PBFE_N1M-16M_E2-10", 
            ani.height = 750, 
            ani.width = 750)
data <- read.csv("Porto_PBFE_N1M-16M_E2-10.csv", header = F)
ylimit = max(data$V5)
saveGIF({
  for(i in c(1,2,4,8)){
    dataset <- paste0(i,"M")
    sample <- data[data$V3 == dataset,c(1,2,5)]
    names(sample) = c("Algorithm","Epsilon", "Time")
    g = ggplot(data=sample, aes(x=factor(Epsilon), y=Time, group=Algorithm, colour=Algorithm, shape=Algorithm)) +
      geom_line(aes(linetype=Algorithm)) +
      geom_point(size=3) +
      labs(title=paste("Execution time in Porto dataset (N=",dataset,").")) + 
      scale_y_continuous("Time (sec)", limits=c(0,ylimit)) +
      scale_x_discrete(expression(paste(epsilon," (mts)"))) 
    plot(g)
  }  
})