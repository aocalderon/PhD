library(data.table)
library(ggplot2)
library(reshape)
library(xtable)

data <- read.csv("S_10P10_P10-30_01.csv", header = F)
data <- rbind(data, read.csv("S_10P10_P10-30_02.csv", header = F))
data <- rbind(data, read.csv("S_10P10_P10-30_03.csv", header = F))
data <- rbind(data, read.csv("S_10P10_P10-30_04.csv", header = F))
data <- rbind(data, read.csv("S_10P10_P10-30_05.csv", header = F))
data <- rbind(data, read.csv("S_10P10_P4-8_01.csv", header = F))
data <- rbind(data, read.csv("S_10P10_P4-8_02.csv", header = F))
data <- rbind(data, read.csv("S_10P10_P4-8_03.csv", header = F))
data <- rbind(data, read.csv("S_10P10_P4-8_04.csv", header = F))
data <- rbind(data, read.csv("S_10P10_P4-8_05.csv", header = F))
data <- data[,c(1,3,5)]
names(data) = c("S","Cores","Time")
data <- data.table(data)
data <- unique(data[, Time:=mean(Time), by=list(S, Cores)])

legend_title = "Optimization"
breaks = c("S0", "S1","S2","S3")
labels = c("Baseline", "Remove evens", "Remove Bcast", "Reorder loops")

g = ggplot(data=data, aes(x=factor(Cores), y=Time, group=S, colour=S, shape=S)) +
    geom_line(aes(linetype=S)) +
    geom_point(size=2.5) +
    labs(x="Number of processors", y="Time (sec)") +
    scale_colour_discrete(name = legend_title, breaks = breaks, labels = labels) +
    scale_shape_discrete(name = legend_title,  breaks = breaks, labels = labels) +
    scale_linetype_discrete(name = legend_title, breaks = breaks, labels = labels)

pdf("plot.pdf", width = 10.5, height = 7.5)
plot(g)
dev.off()

table = cast(data, Cores ~ S)
names(table) = c("Cores", labels)
table = xtable(table, caption = "Comparing the four versions of the Sieve of Erastosthenes.", label = "tab:table", align="cccccc")
addtorow <- list()
addtorow$pos <- list(0)
addtorow$command <- c(paste0("\\textbf{Number of} & \\multicolumn{4}{c}{\\textbf{", legend_title,"}} \\\\\n \\textbf{Cores} & \\textbf{", paste0(labels, collapse = "} & \\textbf{"), "} \\\\\n"))
print(table, include.rownames=F, add.to.row = addtorow, booktabs = T, include.colnames=F, file="tbl_table.tex")