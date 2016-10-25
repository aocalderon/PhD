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

drawSquare <- function(x1, y1, x2, y2,lwd=1,lty=1,col=1){
  lines(c(x1,x1),c(y1,y2),lwd=lwd,lty=lty,col=col) 
  lines(c(x1,x2),c(y2,y2),lwd=lwd,lty=lty,col=col) 
  lines(c(x2,x2),c(y2,y1),lwd=lwd,lty=lty,col=col) 
  lines(c(x2,x1),c(y1,y1),lwd=lwd,lty=lty,col=col)  
}

drawSquare2 <- function(x1, y1, x2, y2,lwd=1,lty=1,col=1){
  rect(x1, y1, x2, y2, col = "lightgrey", lty=2)
}

drawPartitions <- function(){
  lines(c(0,size),c(size/2, size/2),lty=4,col=4,lwd=1)
  lines(c(size/2,size/2),c(0, size),lty=4,col=4,lwd=1)  
}

n = 10
epsilon = 1
r2 = (epsilon/2)^2
size = 20
x = runif(n, 2*epsilon, size/2-2*epsilon)
y = runif(n, 2*epsilon, size/2-2*epsilon)
x = c(x,runif(n, size/2+2*epsilon, size-2*epsilon))
y = c(y,runif(n, size/2+2*epsilon, size-2*epsilon))
x = c(x,runif(n, 2*epsilon, size/2-2*epsilon))
y = c(y,runif(n, size/2+2*epsilon, size-2*epsilon))
x = c(x,runif(n, size/2+2*epsilon, size-2*epsilon))
y = c(y,runif(n, 2*epsilon, size/2-2*epsilon))

xprime = c(4,13, 9.6,10.2, 14.8,6.4,14)
yprime = c(0.5,0.7, 9.7,10.3, 9.7,19.5,9.6)

dim = 5

pdf("d.pdf",width = dim,height = dim, onefile = T)
par(mar=c(0,0,0,0))

plot(1, asp=1, axes = F, , xlab = "", ylab = "", type='n',
     xlim = c(0 - epsilon, size + epsilon), 
     ylim = c(0 - epsilon, size + epsilon))

drawSquare(0,0,size,size,2)

points(x,y,pch=21,col=1,bg=1,cex=.5)
points(xprime,yprime,pch=21,col=1,bg=1,cex=.5)

plot(1, asp=1, axes = F, , xlab = "", ylab = "", type='n',
     xlim = c(0 - epsilon, size + epsilon), 
     ylim = c(0 - epsilon, size + epsilon))

drawSquare(0,0,size,size,2)

points(x,y,pch=21,col=1,bg=1,cex=.5)
points(xprime,yprime,pch=21,col=1,bg=1,cex=.5)

drawPartitions()

plot(1, asp=1, axes = F, , xlab = "", ylab = "", type='n',
     xlim = c(0 - epsilon, size + epsilon), 
     ylim = c(0 - epsilon, size + epsilon))

drawSquare(0,0,size,size,2)

drawPartitions()

drawSquare2(epsilon,epsilon,size/2-epsilon,size/2-epsilon,1,2,1)
drawSquare2(size/2+epsilon,size/2+epsilon,size-epsilon, size-epsilon,1,2,1)
drawSquare2(size/2+epsilon,epsilon,size-epsilon,size/2-epsilon,1,2,1)
drawSquare2(epsilon,size/2+epsilon,size/2-epsilon,size-epsilon,1,2,1)

arrows(size-epsilon+0.1, size-epsilon+0.1,size-0.1,size-0.1,length = 0.05)
arrows(size-0.1,size-0.1,size-epsilon+0.1, size-epsilon+0.1,length = 0.05)
text(size-epsilon+0.2,size-epsilon/2+0.1,expression(epsilon),cex=0.5)

points(x,y,pch=21,col=1,bg=1,cex=.5)
points(xprime,yprime,pch=21,col=1,bg=1,cex=.5)

plot(1, asp=1, axes = F, , xlab = "", ylab = "", type='n',
     xlim = c(0 - epsilon, size + epsilon), 
     ylim = c(0 - epsilon, size + epsilon))

drawSquare(0,0,size,size,2)

drawSquare2(epsilon,epsilon,size/2-epsilon,size/2-epsilon,1,2,1)
drawSquare2(size/2+epsilon,size/2+epsilon,size-epsilon, size-epsilon,1,2,1)
drawSquare2(size/2+epsilon,epsilon,size-epsilon,size/2-epsilon,1,2,1)
drawSquare2(epsilon,size/2+epsilon,size/2-epsilon,size-epsilon,1,2,1)

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
    draw.circle(centers[1], centers[2], epsilon/2, nv = 1000, border = 2, 
                col = NA, lty = 2, lwd = 0.5)
    draw.circle(centers[3], centers[4], epsilon/2, nv = 1000, border = 2, 
                col = NA, lty = 2, lwd = 0.5)    
  }
}

drawPartitions()

points(x,y,pch=21,col=1,bg=1,cex=.5)
points(xprime,yprime,pch=21,col=1,bg=1,cex=.5)

plot(1, asp=1, axes = F, , xlab = "", ylab = "", type='n',
     xlim = c(0 - epsilon, size + epsilon), 
     ylim = c(0 - epsilon, size + epsilon))

drawSquare(0,0,size,size,2)

drawPartitions()

drawSquare2(epsilon,epsilon,size/2-epsilon,size/2-epsilon,1,2,1)
drawSquare2(size/2+epsilon,size/2+epsilon,size-epsilon, size-epsilon,1,2,1)
drawSquare2(size/2+epsilon,epsilon,size-epsilon,size/2-epsilon,1,2,1)
drawSquare2(epsilon,size/2+epsilon,size/2-epsilon,size-epsilon,1,2,1)

points(xprime,yprime,pch=21,col=1,bg=1,cex=.5)

plot(1, asp=1, axes = F, , xlab = "", ylab = "", type='n',
     xlim = c(0 - epsilon, size + epsilon), 
     ylim = c(0 - epsilon, size + epsilon))

drawSquare(0,0,size,size,2)

points(xprime,yprime,pch=21,col=1,bg=1,cex=.5)

plot(1, asp=1, axes = F, , xlab = "", ylab = "", type='n',
     xlim = c(0 - epsilon, size + epsilon), 
     ylim = c(0 - epsilon, size + epsilon))

drawSquare(0,0,size,size,2)

pointset=data.frame(x=xprime,y=yprime)
data = sqldf("SELECT p1.x AS x1, p1.y AS y1, p2.x AS x2, p2.y AS y2 FROM pointset p1 CROSS JOIN pointset p2 ")

for(i in 1:nrow(data)){
  x1 = data[i,1]
  y1 = data[i,2]
  x2 = data[i,3]
  y2 = data[i,4]
  d = distance(x1,y1,x2,y2)
  if(d <= epsilon && d != 0){
    centers = calculateDisk(x1,y1,x2,y2)
    draw.circle(centers[1], centers[2], epsilon/2, nv = 1000, border = 2, 
                col = NA, lty = 2, lwd = 0.5)
    draw.circle(centers[3], centers[4], epsilon/2, nv = 1000, border = 2, 
                col = NA, lty = 2, lwd = 0.5)    
  }
}

points(xprime,yprime,pch=21,col=1,bg=1,cex=.5)

dev.off()