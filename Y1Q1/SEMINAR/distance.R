vectors_3 <- read.table('vectors_small_3.txt', header = F, sep = " ")
vectors_3 <- vectors_3[,1:4]
king <- sqldf("SELECT V2, V3, V4 from vectors_3 WHERE V1 LIKE 'king'")
man <- sqldf("SELECT V2, V3, V4 from vectors_3 WHERE V1 LIKE 'man'")
woman <- sqldf("SELECT V2, V3, V4 from vectors_3 WHERE V1 LIKE 'woman'")
queen <- sqldf("SELECT V2, V3, V4 from vectors_3 WHERE V1 LIKE 'queen'")

x <- (vectors_3[,2] - king[,1])^2
y <- (vectors_3[,3] - king[,2])^2
z <- (vectors_3[,4] - king[,3])^2

d <- sqrt(x + y + z)

plot3d(vectors_3[,2], vectors_3[,3], vectors_3[,4], col = rgb(d / max(d), 0, 0, d / max(d)))
