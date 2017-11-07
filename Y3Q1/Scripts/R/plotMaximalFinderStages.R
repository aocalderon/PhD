if (!require("pacman")) install.packages("pacman")
pacman::p_load(ggplot2, data.table, foreach, sqldf, tidyr, stringr, dplyr)

PATH = "Y3Q1/Scripts/Scaleup/A/"
filename = "Berlin_D20K-80K_E10.0-50.0_M12_P1024_201711052325"
PHD_HOME = Sys.getenv(c("PHD_HOME"))

f = paste0(PHD_HOME,PATH,filename,".times")
lines = read.table(f, header = F, sep = ">")
lines = str_trim(lines[,2])
output = paste0(PHD_HOME,PATH,"Berlin_Stages_D20K-80K_E10.0-50.0_M12_P1024.dat")
cat("Dataset,Cores,Epsilon,Stages,Time", file=output, sep = "\n")
epsilon = 10.0
for(line in lines){
  if(grepl("Running with mu = ", line)){
    temp = str_split_fixed(line, ",", 3)
    cores = str_split_fixed(temp[2], "=", 2)[,2]
    cores = as.numeric(str_trim(cores))
    if(cores == 7) dataset = "20K"
    else if(cores == 14) dataset = "40K"
    else if(cores == 21) dataset = "60K"
    else if(cores == 28) dataset = "80K"    
  }
  if(grepl("Running epsilon = ", line)){
    temp = str_split_fixed(line, "=", 2)
    epsilon = str_split_fixed(temp[2], "iteration", 2)[,1]
    epsilon = as.numeric(str_trim(epsilon))
  }  
  stage = "Initial indexing"
  if(grepl(stage, line)){
    time = str_split_fixed(line, "\\[", 2)
    time = str_split_fixed(time[2], "ms", 2)[,1]
    indexing = as.numeric(str_trim(time))
  }  
  stage = "Self-join"
  if(grepl(stage, line)){
    time = str_split_fixed(line, "\\[", 2)
    time = str_split_fixed(time[2], "ms", 2)[,1]
    time = as.numeric(str_trim(time))
    text = paste0(dataset,",",cores,",",epsilon,",","Initial indexing",",",indexing,"\n")
    cat(text, file=output, append = T)
    text = paste0(dataset,",",cores,",",epsilon,",",stage,",",time,"\n")
    cat(text, file=output, append = T)
  }  
  stage = "Computing disks"
  if(grepl(stage, line)){
    time = str_split_fixed(line, "\\[", 2)
    time = str_split_fixed(time[2], "ms", 2)[,1]
    time = as.numeric(str_trim(time))
    text = paste0(dataset,",",cores,",",epsilon,",",stage,",",time,"\n")
    cat(text, file=output, append = T)
  }
  stage = "Mapping"
  if(grepl(stage, line)){
    time = str_split_fixed(line, "\\[", 2)
    time = str_split_fixed(time[2], "ms", 2)[,1]
    time = as.numeric(str_trim(time))
    text = paste0(dataset,",",cores,",",epsilon,",",stage,",",time,"\n")
    cat(text, file=output, append = T)
  }
  stage = "Filtering less-than-mu disks"
  if(grepl(stage, line)){
    time = str_split_fixed(line, "\\[", 2)
    time = str_split_fixed(time[2], "ms", 2)[,1]
    time = as.numeric(str_trim(time))
    text = paste0(dataset,",",cores,",",epsilon,",","Filtering mu",",",time,"\n")
    cat(text, file=output, append = T)
  }
  stage = "Indexing candidates"
  if(grepl(stage, line)){
    time = str_split_fixed(line, "\\[", 2)
    time = str_split_fixed(time[2], "ms", 2)[,1]
    time = as.numeric(str_trim(time))
    text = paste0(dataset,",",cores,",",epsilon,",",stage,",",time,"\n")
    cat(text, file=output, append = T)
  }
  stage = "Finding maximal disks inside partitions"
  if(grepl(stage, line)){
    time = str_split_fixed(line, "\\[", 2)
    time = str_split_fixed(time[2], "ms", 2)[,1]
    time = as.numeric(str_trim(time))
    text = paste0(dataset,",",cores,",",epsilon,",",stage,",",time,"\n")
    cat(text, file=output, append = T)
  }
  stage = "Finding maximal disks in frame partitions"
  if(grepl(stage, line)){
    time = str_split_fixed(line, "\\[", 2)
    time = str_split_fixed(time[2], "ms", 2)[,1]
    time = as.numeric(str_trim(time))
    text = paste0(dataset,",",cores,",",epsilon,",",stage,",",time,"\n")
    cat(text, file=output, append = T)
  }
}

data = read.csv(output)
data = sqldf("SELECT Dataset, Cores, Epsilon, Stages, AVG(Time) AS Time FROM data GROUP BY Dataset, Cores, Epsilon, Stages")
temp_title = paste("(radius of disk in mts) in Berlin dataset.")
title = substitute(paste("Execution time by stages and ", epsilon) ~ temp_title, list(temp_title = temp_title))
g = ggplot(data=data, aes(x=factor(Epsilon), y=Time, fill=Dataset)) +
  geom_bar(stat="identity", position=position_dodge(width = 0.75), width = 0.75) +
  labs(title=title, y="Time(s)", x=expression(paste(epsilon,"(mts)"))) +
  facet_wrap(~Stages)
plot(g)
