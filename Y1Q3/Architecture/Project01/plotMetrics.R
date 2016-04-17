require(ggplot2)

metric = "sim_CPI"
variables = c("1", "2", "4", "8")
applications = c("cc1", "go", "anagram", "compress95")
title = "Cycles Per Instruction"
xlab = "Number of fetched instructions"

Data <- expand.grid(metric=metric,
                    variable=variables,
                    application=applications)
Data$value = 0

d = read.csv('C:/and/PhD/Code/PhD/Y1Q3/Architecture/Project01/Results/metrics_F.csv')
d = d[complete.cases(d),1:(length(variables)+1)]
record = d[d$metric==metric,2:(length(variables)+1)]
for(i in 1:length(record)){
  Data[Data$metric==metric & 
         Data$application=='cc1' & 
         Data$variable==variables[i],4]=record[i]
}

d = read.csv('C:/and/PhD/Code/PhD/Y1Q3/Architecture/Project01/Results/metrics_A_F.csv')
d = d[complete.cases(d),1:(length(variables)+1)]
record = d[d$metric==metric,2:(length(variables)+1)]
for(i in 1:length(record)){
  Data[Data$metric==metric & 
         Data$application=='anagram' & 
         Data$variable==variables[i],4]=record[i]
}

d = read.csv('C:/and/PhD/Code/PhD/Y1Q3/Architecture/Project01/Results/metrics_G_F.csv')
d = d[complete.cases(d),1:(length(variables)+1)]
record = d[d$metric==metric,2:(length(variables)+1)]
for(i in 1:length(record)){
  Data[Data$metric==metric & 
         Data$application=='go' & 
         Data$variable==variables[i],4]=record[i]
}

d = read.csv('C:/and/PhD/Code/PhD/Y1Q3/Architecture/Project01/Results/metrics_C_F.csv')
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