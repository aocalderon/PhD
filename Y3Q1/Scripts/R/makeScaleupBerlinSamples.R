#!/usr/bin/Rscript

PHD_HOME = Sys.getenv(c("PHD_HOME"))
PATH = "Y3Q1/Datasets/"
filename = paste0(PHD_HOME,PATH,'B80K.csv')
B80K = read.csv(filename, header = F)
N =nrow(B80K)
i1 = seq(1,N,4)
i2 = seq(2,N,4)
i3 = seq(3,N,4)
B20K = B80K[i1,]
B40K = B80K[sort(c(i1,i2)),]
B60K = B80K[sort(c(i1,i2,i3)),]
write.table(B20K,paste0(PHD_HOME,PATH,'B20K.csv'),row.names = F, col.names = F, sep = ',', quote = F)
write.table(B40K,paste0(PHD_HOME,PATH,'B40K.csv'),row.names = F, col.names = F, sep = ',', quote = F)
write.table(B60K,paste0(PHD_HOME,PATH,'B50K.csv'),row.names = F, col.names = F, sep = ',', quote = F)