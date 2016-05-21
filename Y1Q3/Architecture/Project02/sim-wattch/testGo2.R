interval = 10000
increment = 0.1
app = "go"
turnoff = FALSE
tot_powers1 = c()

targets = seq(40000,80000,2500)
for(target in targets){
  runSim_OutOrder(app,interval,target,increment,turnoff,'output1')
  tot_powers1 = c(tot_powers1, getTotalPower('output1'))
  print('*')
}

turnoff = TRUE
runSim_OutOrder(app,interval,target,increment,turnoff,'output2')
baseline = getTotalPower('output2')

plot(targets, tot_powers,type = 'l',col=4)
abline(h=baseline,col=2,lty=2)
