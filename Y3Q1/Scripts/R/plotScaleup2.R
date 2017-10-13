if (!require("pacman")) install.packages("pacman")
pacman::p_load(ggplot2, data.table, foreach, sqldf)

PATH = "Y3Q1/Scripts/Scaleup/"
filename ="Berlin_N20K-80K_E50.0-100.0"
PHD_HOME = Sys.getenv(c("PHD_HOME"))

files = system(paste0("ls ",PHD_HOME,PATH,filename,"_C*.csv"), intern = T)
data = data.frame()
foreach(f = files) %do% {
  data = rbind(data, read.csv(f, header = F))
}
data = data[, c(2, 3, 6, 9)]
names(data) = c("Epsilon", "Dataset", "Time", "Cores")
data20 = sqldf("SELECT Epsilon, Dataset, Cores, Time FROM data WHERE Dataset LIKE '20K' AND Cores = 7")
data40 = sqldf("SELECT Epsilon, Dataset, Cores, Time FROM data WHERE Dataset LIKE '40K' AND Cores = 14")
data60 = sqldf("SELECT Epsilon, Dataset, Cores, Time FROM data WHERE Dataset LIKE '60K' AND Cores = 21")
data80 = sqldf("SELECT Epsilon, Dataset, Cores, Time FROM data WHERE Dataset LIKE '80K' AND Cores = 28")
data = rbind(data20, data40, data60, data80)
data = sqldf("SELECT Epsilon, Dataset, Cores, AVG(Time) AS Time FROM data GROUP BY 1, 2, 3")
data$Cores = factor(data$Cores)

temp_title = paste("(radius of disk in mts) in Berlin dataset.")
title = substitute(paste("Scaleup by ", epsilon) ~ temp_title, list(temp_title = temp_title))
g = ggplot(data=data, aes(x=factor(Epsilon), y=Time, fill=Cores)) +
  geom_bar(stat="identity", position=position_dodge(width = 0.75),width = 0.75) +
  labs(title=title, y="Time(s)", x=expression(paste(epsilon,"(mts)")))
plot(g)

#####
# Plotting number of candidates and maximal disks...
#####

## Second_Scaleup
PATH = "Y3Q1/Scripts/Scaleup/"
filename ="Berlin_N20K-80K_E50.0-100.0"
files = system(paste0("ls ",PHD_HOME,PATH,filename,"_C*.csv"), intern = T)
data = data.frame()
foreach(f = files) %do% {
  data = rbind(data, read.csv(f, header = F))
}
data = data[, c(2, 3, 7, 8, 9)]
names(data) = c("Epsilon", "Dataset", "Candidates", "Maximals", "Cores")
data20 = sqldf("SELECT Epsilon, Dataset, Cores, Candidates, Maximals FROM data WHERE Dataset LIKE '20K' AND Cores = 7")
data40 = sqldf("SELECT Epsilon, Dataset, Cores, Candidates, Maximals FROM data WHERE Dataset LIKE '40K' AND Cores = 14")
data60 = sqldf("SELECT Epsilon, Dataset, Cores, Candidates, Maximals FROM data WHERE Dataset LIKE '60K' AND Cores = 21")
data80 = sqldf("SELECT Epsilon, Dataset, Cores, Candidates, Maximals FROM data WHERE Dataset LIKE '80K' AND Cores = 28")
data = rbind(data20, data40, data60, data80)
data = sqldf("SELECT DISTINCT Epsilon, Dataset, Cores, Candidates, Maximals FROM data")
data$Dataset = factor(data$Dataset, levels = paste0(seq(20, 80, 20), "K"))
data$Cores = factor(data$Cores)

library("tidyr")
data = gather(data, "Disks", "Count", 4:5)
candidates2 = data[data$Disks == "Candidates",]
temp_title = paste("(radius of disk in mts) in Berlin dataset.")
title = substitute(paste("Number of Candidate disks by ", epsilon) ~ temp_title, list(temp_title = temp_title))
g = ggplot(data=candidates2, aes(x=factor(Dataset), y=Count, group = Disks, colour = Disks, shape = Disks)) + 
  geom_line(aes(linetype = Disks)) + 
  geom_point(size = 2) + 
  labs(title = title, y = "Count") + 
  scale_x_discrete("Dataset") +
  theme(axis.text.x = element_text(size = 8, angle = 90), axis.text.y = element_text(size = 8)) + 
  facet_wrap(~Epsilon) + 
  scale_color_manual(values=c("#F8766D")) + 
  scale_shape_manual(values=c(16)) + 
  scale_linetype_discrete()
plot(g)
ggsave("Candidates_1.png", width = 20, height = 14, units = "cm")

maximals2 = data[data$Disks == "Maximals",]
title = substitute(paste("Number of Maximal disks by ", epsilon) ~ temp_title, list(temp_title = temp_title))
g = ggplot(data=maximals2, aes(x=factor(Dataset), y=Count, group = Disks, colour = Disks, shape = Disks)) + 
  geom_line(aes(linetype = Disks)) + 
  geom_point(size = 2) + 
  labs(title = title, y = "Count") + 
  scale_x_discrete("Dataset") +
  theme(axis.text.x = element_text(size = 8, angle = 90), axis.text.y = element_text(size = 8)) + 
  facet_wrap(~Epsilon) + 
  scale_color_manual(values=c("#00BFC4")) + 
  scale_shape_manual(values=c(16)) + 
  scale_linetype_discrete()
plot(g)
ggsave("Maximals_1.png", width = 20, height = 14, units = "cm")

