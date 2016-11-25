library(animation)
library(ggplot2)

#Set delay between frames when replaying
ani.options(interval=0.25,
            movie.name = "Beijing10K-100K_E10-200", 
            ani.height = 750, 
            ani.width = 750)
data <- read.csv("Beijing10K-100K_E10-200.csv", header = F)

# Begin animation loop
# Note the brackets within the parentheses
saveGIF({
  for(i in seq(10,100,10)){
    dataset <- paste0(i,"K")
    sample <- data[data$V3 == dataset,c(1,2,5)]
    names(sample) = c("Algorithm","Epsilon", "Time")
    g = ggplot(data=sample, aes(x=factor(Epsilon), y=Time, group=Algorithm, colour=Algorithm, shape=Algorithm)) +
      geom_line(aes(linetype=Algorithm)) +
      geom_point(size=2.5) +
      labs(title=paste("Execution time in Beijing dataset (N=",dataset,").")) + 
      scale_y_continuous("Time (sec)", limits=c(0,300)) +
      scale_x_discrete(expression(paste(epsilon," (mts)")),
                       breaks=as.character(seq(20,200,20))) 
    plot(g)
  }  
})