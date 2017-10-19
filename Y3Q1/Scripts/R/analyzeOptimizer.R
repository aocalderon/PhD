if (!require("pacman")) install.packages("pacman")
pacman::p_load(stringr, plotly)

log = read.table('~/PhD/Y3Q1/Scripts/Misc/optimizer_output.log')
data = as.data.frame(str_split_fixed(as.character(log$V7),",",10))
names(data) = c('Dataset','size','Partitions','Entries','time','avg','sd','var','min','max')
data$size = as.numeric(data$size)
data$Partitions = as.numeric(data$Partitions)
data$Entries = as.numeric(data$Entries)
data$time = as.numeric(data$time)
data$avg = as.numeric(data$avg)
data$sd = as.numeric(data$sd)
data$var = as.numeric(data$var)
data$min = as.numeric(data$min)
data$max = as.numeric(data$max)
p <- plot_ly(data = data, x = ~Partitions, y = ~Entries, z = ~time, color = ~Dataset) %>%
  add_markers() %>%
  layout(scene = list(xaxis = list(title = 'Partitions'),
                      yaxis = list(title = 'Maximun entries per node'),
                      zaxis = list(title = 'Time (s)')))
p