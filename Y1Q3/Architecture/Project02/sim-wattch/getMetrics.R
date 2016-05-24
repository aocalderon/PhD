getPowerMetrics <- function(filename){
  dd <- readLines(filename)
  start <- match("sim: ** simulation statistics **", dd)
  end <- start + 282
  dd <- dd[start:end]
  n <- length(dd)
  empty <- rep(0, n)
  results <- data.frame(metric = empty, value = empty)
  for(i in 1:n){
    result <- strsplit(strsplit(dd[i], "#")[[1]][1], "\\s+")[[1]]
    results[i,1] <- result[1]
    results[i,2] <- as.numeric(result[2])
  }
  results <- results[complete.cases(results),]  
  
  return(results[grep("power", results$metric), ])
}

getTotalPower <- function(filename){
  dd <- readLines(filename)
  start <- match("sim: ** simulation statistics **", dd)
  end <- start + 282
  dd <- dd[start:end]
  line <- dd[grep("tot_power",dd)]
  result <- strsplit(strsplit(line, "#")[[1]][1], "\\s+")[[1]]
  
  return(as.numeric(result[2]))
}

runSim_OutOrder <- function(app, interval, target, increment = 0.1, turnoff = FALSE, output){
  command = ""
  command = paste0(command, "./sim-outorder -max:inst 10000000") 
  command = paste0(command, " -DVFSInterval ",format(interval,scientific=F))
  command = paste0(command, " -DVFSTargetPower ",format(target,scientific=F))
  command = paste0(command, " -DVFSTurnOff ",turnoff)
  command = paste0(command, " -DVFSIncrement ",increment)
  command = paste0(command, " ../benchmarks/",app,".alpha 50 9 ../benchmarks/2stone9.in")
  command = paste0(command, " 2&> ", output)
  system(command, wait = TRUE)
  r = read.table('test.txt', sep = ':')
  
  return(r)
}

runSim_OutOrderGo <- function(interval, target, increment = 0.1, turnoff = FALSE, output){
  command = ""
  command = paste0(command, "./sim-outorder -max:inst 50000000") 
  command = paste0(command, " -DVFSInterval ",format(interval,scientific=F))
  command = paste0(command, " -DVFSTargetPower ",format(target,scientific=F))
  command = paste0(command, " -DVFSTurnOff ",turnoff)
  #command = paste0(command, " -DVFSIncrement ",increment)
  command = paste0(command, " ../benchmarks/go.alpha 50 9 ../benchmarks/2stone9.in")
  command = paste0(command, " 2&> ", output)
  system(command, wait = TRUE)
  r = read.table('test.txt', sep = ':')
  
  return(r)
}

runSim_OutOrderAnagram <- function(interval, target, increment = 0.2, turnoff = FALSE, output){
  command = ""
  command = paste0(command, "./sim-outorder ") 
  command = paste0(command, " -DVFSInterval ",format(interval,scientific=F))
  command = paste0(command, " -DVFSTargetPower ",format(target,scientific=F))
  command = paste0(command, " -DVFSTurnOff ",turnoff)
  #command = paste0(command, " -DVFSIncrement ",increment)
  command = paste0(command, " ../benchmarks/anagram.alpha ../benchmarks/words < ../benchmarks/anagram.in")
  command = paste0(command, " 2&> ", output)
  system(command, wait = TRUE)
  r = read.table('test.txt', sep = ':')
  
  return(r)
}

plotIntervals <- function(dd, target){
  plot(1:nrow(dd), dd$V1, type = 'l', xlab='Iteration', ylab='Total Power per Cycle',col=4)
  # abline(h=mean(dd$V1), col=4, lwd=1, lty=2)
  abline(h=target, col=2, lwd=1, lty=2)
}