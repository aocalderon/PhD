require(ggplot2)

plotMetric <- function(feature, metric, variables, applications, xlab){
  metrics = read.csv('metrics.csv')
  title = simpleCap(as.character(metrics$desc[metrics$metric==metric]))
  Data <- expand.grid(metric=metric,
                      variable=variables,
                      application=applications)
  Data$value = 0
  
  d = read.csv(paste0('Results/metrics',feature,'.csv'))
  d = d[complete.cases(d),1:(length(variables)+1)]
  record = d[d$metric==metric,2:(length(variables)+1)]
  for(i in 1:length(record)){
    Data[Data$metric==metric & 
           Data$application=='cc1' & 
           Data$variable==variables[i],4]=record[i]
  }
  
  d = read.csv(paste0('Results/metrics_A',feature,'.csv'))
  d = d[complete.cases(d),1:(length(variables)+1)]
  record = d[d$metric==metric,2:(length(variables)+1)]
  for(i in 1:length(record)){
    Data[Data$metric==metric & 
           Data$application=='anagram' & 
           Data$variable==variables[i],4]=record[i]
  }
  
  d = read.csv(paste0('Results/metrics_G',feature,'.csv'))
  d = d[complete.cases(d),1:(length(variables)+1)]
  record = d[d$metric==metric,2:(length(variables)+1)]
  for(i in 1:length(record)){
    Data[Data$metric==metric & 
           Data$application=='go' & 
           Data$variable==variables[i],4]=record[i]
  }
  
  d = read.csv(paste0('Results/metrics_C',feature,'.csv'))
  d = d[complete.cases(d),1:(length(variables)+1)]
  record = d[d$metric==metric,2:(length(variables)+1)]
  for(i in 1:length(record)){
    Data[Data$metric==metric & 
           Data$application=='compress95' & 
           Data$variable==variables[i],4]=record[i]
  }
  
  g <- ggplot(data=Data, 
              aes(x=factor(variable), y=value, 
                  group=metric,
                  shape=metric,
                  color=metric)) + 
    geom_line() + 
    geom_point() +
    labs(title = title) +
    scale_x_discrete(xlab) +
    scale_y_continuous(metric) + 
    guides(color=F, shape=F)  +
    facet_grid(.~application)
  
  plot(g)
  ggsave(paste0('Plots/plot',feature,'_',metric,'.pdf'))
}

simpleCap <- function(x) {
  s <- strsplit(x, " ")[[1]]
  paste(toupper(substring(s, 1,1)), substring(s, 2),
        sep="", collapse=" ")
}