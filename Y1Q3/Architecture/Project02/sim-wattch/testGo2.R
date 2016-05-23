interval = 100000
increment = 0.2
turnoff = FALSE
tot_powers = c()

targets = seq(5000000,7000000,100000)
n = length(targets)
i = 1
for(target in targets){
  runSim_OutOrderGo(interval,target,increment,turnoff,'output1')
  tot_powers = c(tot_powers, getTotalPower('output1'))
  print(paste0('* ',i,'/',n))
  i = i + 1
}

turnoff = TRUE
runSim_OutOrderGo(interval,target,increment,turnoff,'output2')
baseline = getTotalPower('output2')

plot(targets, tot_powers,type = 'l',col=4)
abline(h=baseline,col=2,lty=2)
