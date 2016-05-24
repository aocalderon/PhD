source('getMetrics.R')
interval = 100000
target = 5400000
increment = 0.2

turnoff = FALSE
output = 'output1'
DVFSOn = runSim_OutOrderGo(interval,target,increment,turnoff,output)
plotIntervals(DVFSOn, target)
output1 = getPowerMetrics('output1')

turnoff = TRUE
output = 'output2'
DVFSOff = runSim_OutOrderGo(interval,target,increment,turnoff,output)
plotIntervals(DVFSOff, target)
output2 = getPowerMetrics('output2')

print(paste0("DVFS Controller ON"))
print(summary(DVFSOn))
print(paste0("DVFS Controller OFF"))
print(summary(DVFSOff))

print(getTotalPower('output1'))
print(getTotalPower('output2'))

data = merge(output1, output2, by='metric')
names(data) = c('metric','DVFS','Baseline')
data$diff = ((data$Baseline-data$DVFS)/data$Baseline)*100
data = data[with(data, order(-diff)),]