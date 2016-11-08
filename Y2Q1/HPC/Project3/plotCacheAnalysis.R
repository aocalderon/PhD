library(data.table)
library(ggplot2)
library(reshape)
library(xtable)
library(plotrix)

data <- read.csv("cache2.csv", header = F)
data <- data[,c(1,2,6)]
names(data) = c("CO","CP","Time")
data <- data.table(data)
data <- unique(data[, Time:=mean(Time), by=list(CO, CP)])

# legend_title = "Optimization"
# breaks = c("S0", "S1","S2","S3")
# labels = c("Baseline", "Remove evens", "Remove Bcast", "Reorder loops")
# 
# g = ggplot(data=data, aes(x=factor(Cores), y=Time, group=S, colour=S, shape=S)) +
#   geom_line(aes(linetype=S)) +
#   geom_point(size=2.5) +
#   labs(x="Number of processors", y="Time (sec)") +
#   scale_colour_discrete(name = legend_title, breaks = breaks, labels = labels) +
#   scale_shape_discrete(name = legend_title,  breaks = breaks, labels = labels) +
#   scale_linetype_discrete(name = legend_title, breaks = breaks, labels = labels)
# 
# pdf("plot2.pdf", width = 10.5, height = 7.5)
# plot(g)
# dev.off()

table = cast(data, CO ~ CP)
# names(table) = c("Cache", labels)
# table[4,5] = NA
table = table[3:5,]
pdf("cache2.pdf", width = 7, height = 5)
color2D.matplot(table[,2:ncol(table)], 
                show.values = 6,
                axes = F,
                xlab = "Block size for the prime array",
                ylab = "Block size for the odds array",
                vcex = 0.75,
                vcol = "#444444",
                yrev = T,
                na.color = "green",
                extremes = c("#00DD00","yellow", "red"))
axis(1, at = seq_len(ncol(table) - 1) - 0.5, labels = names(table)[2:ncol(table)], tick = FALSE, cex.axis = 0.75)
axis(2, at = seq_len(nrow(table)) - 0.5, labels = rev(table$CO), tick = FALSE, las = 1, cex.axis = 0.75)
# textbox(c(3.15,4), 1.55, c("0.379629"), box = F, cex=.85, col="black")
dev.off()
# table = xtable(table, caption = "Comparing the different values of cache size for odds and prime arrays.", label = "tab:table3", align="cccccc")
# addtorow <- list()
# addtorow$pos <- list(0)
# addtorow$command <- c(paste0("\\textbf{Number of} & \\multicolumn{4}{c}{\\textbf{", legend_title,"}} \\\\\n \\textbf{Cores} & \\textbf{", paste0(labels, collapse = "} & \\textbf{"), "} \\\\\n"))
# print(table, include.rownames=F, add.to.row = addtorow, booktabs = T, include.colnames=F, file="tbl_table3.tex")