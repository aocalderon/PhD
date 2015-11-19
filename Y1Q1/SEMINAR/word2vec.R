library("GGally", lib.loc="~/R/x86_64-pc-linux-gnu-library/3.2")
library("ggplot2", lib.loc="~/R/x86_64-pc-linux-gnu-library/3.2")
library("sqldf")

setwd('/home/and/Documents/Projects/C++/word2vec/trunk/')
# vectors <- read.table('vectors.txt', header = F, sep = " ")
dims <- dim(vectors)[2] - 1
x <- sample(nrow(vectors), 200)
y <- seq(1, dims, 5)
v <- vectors[x, y]

king <- sqldf("SELECT * FROM vectors WHERE V1 LIKE 'king'")
man <- sqldf("SELECT * FROM vectors WHERE V1 LIKE 'man'")
woman <- sqldf("SELECT * FROM vectors WHERE V1 LIKE 'woman'")
queen <- sqldf("SELECT* FROM vectors WHERE V1 LIKE 'queen'")

y <- seq(1, dims, 4)
v <- vectors[x, y]
v$V1 <- ''
q <- rbind(king[, y],man[, y],woman[, y],queen[, y], v)
q$alpha <- 1.0
q[q$V1=='','alpha'] <- 0.15
q$size <- 1.0
q[q$V1=='','size'] <- 0.5
ggparcoord(data = q, columns = 2:length(y), showPoints = F, groupColumn = 1, alphaLines = 'alpha', mapping = ggplot2::aes(size = 0.6)) + scale_alpha(guide = 'none') + theme(axis.text.x = element_text(angle = 90))

# 
# x <- (vectors_3[,2] - king[,1])^2
# y <- (vectors_3[,3] - king[,2])^2
# z <- (vectors_3[,4] - king[,3])^2
# 
# d <- sqrt(x + y + z)