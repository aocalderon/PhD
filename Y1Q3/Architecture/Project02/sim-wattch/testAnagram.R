source('getMetrics.R')
interval = 100000
target = 7425842
increment = 0.2

turnoff = FALSE
output = 'output3'
DVFSOn = runSim_OutOrderAnagram(interval,target,increment,turnoff,output)
plotIntervals(DVFSOn, target)
output1 = getPowerMetrics('output3')

turnoff = TRUE
output = 'output4'
DVFSOff = runSim_OutOrderAnagram(interval,target,increment,turnoff,output)
plotIntervals(DVFSOff, target)
output2 = getPowerMetrics('output4')

print(paste0("DVFS Controller ON"))
print(summary(DVFSOn))
print(paste0("DVFS Controller OFF"))
print(summary(DVFSOff))

print(getTotalPower('output3'))
print(getTotalPower('output4'))

data = merge(output1, output2, by='metric')
names(data) = c('metric','DVFS','Baseline')
data$diff = ((data$Baseline-data$DVFS)/data$Baseline)*100
data = data[with(data, order(-diff)),]
