source('plotMetrics.R')

metric = "sim_CPI"
variables = c("Not Taken", "Taken", "Bimodal", "2-Level")
applications = c("cc1", "go", "anagram", "compress95")
feature = "_BP"
xlab = "Branch prediction"

plotMetric(feature, metric, variables, applications, xlab)

cc1 = read.csv(paste0('Results/metrics',feature,'.csv'))
cc1 = cc1[complete.cases(cc1),c(1,length(cc1))]
go = read.csv(paste0('Results/metrics_G',feature,'.csv'))
go = go[complete.cases(go),c(1,length(go))]
features = merge(cc1, go, by = "metric")
names(features) = c('metric','cc1','go')

anagram = read.csv(paste0('Results/metrics_A',feature,'.csv'))
anagram = anagram[complete.cases(anagram),c(1,length(anagram))]
features = merge(features, anagram, by = "metric")
names(features) = c('metric','cc1','go','anagram')

compress95 = read.csv(paste0('Results/metrics_C',feature,'.csv'))
compress95 = compress95[complete.cases(compress95),c(1,length(compress95))]
features = merge(features, compress95, by = "metric")
names(features) = c('metric','cc1','go','anagram','compress95')

features$index = apply(features[2:5],1,sum)
features <- features[with(features,order(-index)),]

for(metric in features$metric[1:50]){
  plotMetric(feature, metric, variables, applications, xlab)
}
