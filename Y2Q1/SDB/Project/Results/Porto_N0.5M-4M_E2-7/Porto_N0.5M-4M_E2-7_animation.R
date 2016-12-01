library(animation)
library(ggplot2)

ani.options(interval=0.25,
            movie.name = "Porto_N0.5M-4M_E2-7", 
            ani.height = 750, 
            ani.width = 750)
data <- read.csv("Porto_N0.5M-4M_E2-7.csv", header = F)
saveGIF({
  for(i in seq(0.5,4,0.5)){
    dataset <- paste0(i,"M")
    sample <- data[data$V3 == dataset,c(1,2,5)]
    names(sample) = c("Algorithm","Epsilon", "Time")
    g = ggplot(data=sample, aes(x=factor(Epsilon), y=Time, group=Algorithm, colour=Algorithm, shape=Algorithm)) +
      geom_line(aes(linetype=Algorithm)) +
      geom_point(size=3) +
      labs(title=paste("Execution time in Porto dataset (N=",dataset,").")) + 
      scale_y_continuous("Time (sec)", limits=c(0,900)) +
      scale_x_discrete(expression(paste(epsilon," (mts)"))) 
    plot(g)
  }  
})