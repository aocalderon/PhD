#!/usr/bin/Rscript

if (!require("pacman")) install.packages("pacman")
pacman::p_load(sqldf, stringr)

runLogETL <- function(PHD_HOME,PHD_PATH,FILENAME,EXTENSION){
  url = paste0(PHD_HOME,PHD_PATH,FILENAME,".",EXTENSION)
  
  lines = readLines(url)
  data = c()
  stages = c()
  for(line in lines){
    if(grepl("B*0K,", line)){
      data = c(data, line)
    }
    if(grepl("\\> \\d", line, perl = T)){
      stages = c(stages, line)
    }
  }
  data = str_split_fixed(data, ">", 2)
  data = as.data.frame(str_split_fixed(data[,2], ",", 9))
  names(data) = c("Dataset", "Epsilon", "Cores", "Mu", "Time", "NPairs", "NCenters", "NDisks", "NMaximals")
  write.table(sapply(data, trimws, which="both"), paste0(PHD_HOME,PHD_PATH,FILENAME,".csv"), col.names = T, row.names = F, sep = ",", quote = F)
  stages = as.data.frame(str_split_fixed(stages, "->", 2)[,2])
  write.table(sapply(stages, trimws, which="both"), paste0(PHD_HOME,PHD_PATH,FILENAME,".txt"), col.names = F, row.names = F, quote = F)
}

#PHD_HOME = Sys.getenv(c("PHD_HOME"))
#PHD_PATH = "Y3Q1/Scripts/Scaleup/"
#FILENAME = "Berlin_N20K-40K_E10-50_M10_C7-14_2017-11-30_11-11"
#EXTENSION = "out"

#runLogETL(PHD_HOME, PHD_PATH, FILENAME, EXTENSION)