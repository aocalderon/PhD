library("sqldf", lib.loc="~/R/x86_64-pc-linux-gnu-library/3.2")
library("plotrix", lib.loc="~/R/x86_64-pc-linux-gnu-library/3.2")

distance <- function(x1, y1, x2, y2){
  return(sqrt((x1-x2)^2+(y1-y2)^2))
}

calculateDisk <- function(x1, y1, x2, y2){
  x = x1 - x2
  y = y1 - y2
  d2 = x^2 + y^2
  if(d2 != 0){
    root = sqrt(abs(4 * (r2 / d2) - 1))
    h1 = ((x + y * root) / 2) + x2
    h2 = ((x - y * root) / 2) + x2
    k1 = ((y - x * root) / 2) + y2
    k2 = ((y + x * root) / 2) + y2
    return(c(h1, k1, h2, k2))
  }
  return(NULL)
}

n = 200
epsilon = n / 10
r2 = (epsilon / 2)^2
size = 3

x = runif(n, 0, n)
y = runif(n, 0, n)

par(mar=c(0.1,0.1,0.1,0.1))
pointset=data.frame(x=x,y=y)
data = sqldf("SELECT p1.x AS x1, p1.y AS y1, p2.x AS x2, p2.y AS y2 FROM pointset p1 CROSS JOIN pointset p2 ")
plot(1, asp=1, axes = F, xlim = c(0 - epsilon, n + epsilon), ylim = c(0 - epsilon, n + epsilon), xlab = "", ylab = "", type='n')
for(i in 1:nrow(data)){
  x1 = data[i,1]
  y1 = data[i,2]
  x2 = data[i,3]
  y2 = data[i,4]
  d = distance(x1,y1,x2,y2)
  if(d <= epsilon && d != 0){
    centers = calculateDisk(x1,y1,x2,y2)
    draw.circle(centers[1], centers[2], epsilon/2, nv = 1000, border = 2, col = NA, lty = 2, lwd = 0.5)
    draw.circle(centers[3], centers[4], epsilon/2, nv = 1000, border = 2, col = NA, lty = 2, lwd = 0.5)    
  }
}
points(x, y, pch = 21, cex = 0.5, col = 1, bg = 1)
box()

pdf("test.pdf",width=size,height=size)
par(mar=c(0.1,0.1,0.1,0.1))
plot(1, asp=1, axes = F, xlim = c(0 - epsilon, n + epsilon), ylim = c(0 - epsilon, n + epsilon), xlab = "", ylab = "", type='n')
pointset=data.frame(x=x,y=y)
data = sqldf("SELECT p1.x AS x1, p1.y AS y1, p2.x AS x2, p2.y AS y2 FROM pointset p1 CROSS JOIN pointset p2 ")

for(i in 1:nrow(data)){
  x1 = data[i,1]
  y1 = data[i,2]
  x2 = data[i,3]
  y2 = data[i,4]
  d = distance(x1,y1,x2,y2)
  if(d <= epsilon && d != 0){
    centers = calculateDisk(x1,y1,x2,y2)
    draw.circle(centers[1], centers[2], epsilon/2, nv = 1000, border = 2, col = NA, lty = 2, lwd = 0.5)
    draw.circle(centers[3], centers[4], epsilon/2, nv = 1000, border = 2, col = NA, lty = 2, lwd = 0.5)    
  }
}
points(x, y, pch = 21, cex = 0.5, col = 1, bg = 1 )
box()
dev.off()