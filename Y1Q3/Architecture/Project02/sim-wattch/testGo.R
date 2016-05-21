interval = 10000
target = 500000
increment = 0.1
app = "go"

turnoff = FALSE
DVFSOn = runSim_OutOrder(app,interval,target,increment,turnoff,'output1')
plotIntervals(DVFSOn, target)
output1 = getPowerMetrics('output1')

turnoff = TRUE
DVFSOff = runSim_OutOrder(app,interval,target,increment,turnoff,'output2')
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