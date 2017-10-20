if (!require("pacman")) install.packages("pacman")
pacman::p_load(stringr, plotly, dplyr, reshape2)

###################
# Setting credentials...
###################
Sys.setenv("plotly_username"="aocalderon1978")
Sys.setenv("plotly_api_key"="dx4LIeqcXzokLrO2SUHF")
###################
# Setting variables...
###################
PHD_HOME = Sys.getenv(c("PHD_HOME"))
PATH = "Y3Q1/Scripts/Misc/"
EXTENSION = ".txt"
LOGNAME = "OO_D40K-80K_En2-26_P512-3072_E50_M12"
filename = paste0(PHD_HOME,PATH,LOGNAME,EXTENSION)
log = read.table(filename)
data = as.data.frame(str_split_fixed(as.character(log$V7),",",14), stringsAsFactors = F)
names(data) = c('tag','Dataset','NCandidates','Partitions1','Partitions2','Entries','Epsilon','Mu','time','avg','sd','var','min','max')
data$NCandidates = as.numeric(data$NCandidates)
data$Partitions1 = as.numeric(data$Partitions1)
data$Partitions2 = as.numeric(data$Partitions2)
data$Entries = as.numeric(data$Entries)
data$Epsilon = as.numeric(data$Epsilon)
data$Mu = as.numeric(data$Mu)
data$time = as.numeric(data$time)
data$avg = as.numeric(data$avg)
data$sd = as.numeric(data$sd)
data$var = as.numeric(data$var)
data$min = as.numeric(data$min)
data$max = as.numeric(data$max)

# p <- plot_ly(data = data, x = ~Partitions1, y = ~Entries, z = ~time, color = ~Dataset) %>% add_markers() %>%
# layout(scene = list(xaxis = list(title = 'Partitions'), yaxis = list(title = 'Maximun entries per node'), zaxis = list(title = 'Time (s)')))

plotMetric <- function(data, metric){
  metricData = data[,c('Dataset', 'Partitions1', 'Entries', metric)]
  dataSurface = acast(metricData, Partitions1 ~ Entries ~ Dataset)
  tvx = colnames(dataSurface[,,1])
  tvy = rownames(dataSurface[,,1])
  axx <- list(title = 'Maximun entries per node', tickmode = "array", tickvals = 0:4, ticktext = tvx)
  axy <- list(title = 'Partitions', tickmode = "array", tickvals = 0:5, ticktext = tvy)
  axz <- list(title = metric)
  p <- plot_ly(showscale = F) %>%
    add_surface(z = ~dataSurface[,,3]) %>%
    add_surface(z = ~dataSurface[,,2]) %>%
    add_surface(z = ~dataSurface[,,1])  %>%
    layout(scene = list(xaxis = axx, yaxis = axy, zaxis = axz))
           
  chart_link = api_create(p, filename=metric, fileopt = "overwrite")
  return(chart_link)
}

timeLink = plotMetric(data, 'time')
maxLink = plotMetric(data, 'max')