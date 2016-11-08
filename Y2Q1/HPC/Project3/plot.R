library(data.table)
library(ggplot2)
library(reshape)
library(xtable)

data <- read.csv("s.csv", header = F)
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

#pdf("plot.pdf", width = 10.5, height = 7.5)
plot(g)
#dev.off()

table = cast(data, Cores ~ S)
names(table) = c("Cores", labels)

color2D.matplot(table[,2:5], 
                show.values = T,
                axes = F,
                xlab = "",
                ylab = "",
                vcex = 0.75,
                vcol = "black",
                extremes = c("green","yellow", "red"))
axis(3, at = seq_len(ncol(table) - 1) - 0.5, labels = labels, tick = FALSE, cex.axis = 0.75)
axis(2, at = seq_len(nrow(table)) - 0.5, labels = rev(table$Cores), tick = FALSE, las = 1, cex.axis = 0.75)

table = xtable(table, caption = "Comparing the four versions of the Sieve of Erastosthenes.", label = "tab:table1", align="cccccc")
addtorow <- list()
addtorow$pos <- list(0)
addtorow$command <- c(paste0("\\textbf{Number of} & \\multicolumn{4}{c}{\\textbf{", legend_title,"}} \\\\\n \\textbf{Cores} & \\textbf{", paste0(labels, collapse = "} & \\textbf{"), "} \\\\\n"))
print(table, include.rownames=F, add.to.row = addtorow, booktabs = T, include.colnames=F, file="tbl_table.tex")