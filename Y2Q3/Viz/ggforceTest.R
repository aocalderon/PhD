library(ggforce)

epsilon = 100
points = read.csv('p_4799.csv', header = F)
names(points) = c("ID","lng","lat")
points = points[points$lng > -326499.3,]
points$ID = 1:nrow(points)
f0 = read.csv('f0_4799.csv', header = F)
names(f0) = c("ID","lng","lat")
f0 = f0[f0$lng > -326499.3,]
f0$ID = 1:nrow(f0)

g = ggplot() + 
  coord_fixed() + 
  geom_circle(aes(x0 = lng, y0 = lat, r = epsilon/2), col = 'red',
              bg = adjustcolor('red', alpha = 0.03), data=f0, lty = 2, lwd = 0.25) +
  geom_point(aes(x=lng, y=lat), data = points, colour = 'blue') +
  geom_text_repel(aes(lng, lat, label = ID), fontface = 'bold', data = points) +
  geom_point(aes(x=lng - 35.35, y=lat + 35.35), data = f0[f0$ID >= 8,], size = 1) +
  geom_text_repel(aes(lng - 35.35, lat + 35.35, label = paste0("C",ID)), 
                  fontface = 'bold', color = 'red', size = 3,
                  box.padding = unit(0.5, "lines"),
                  point.padding = unit(0.1, "lines"),
                  segment.color = 'black',
                  data = f0[f0$ID >= 8,]) +
  geom_point(aes(x=lng + 35.35, y=lat + 35.35), data = f0[f0$ID < 8,], size = 1) +
  geom_text_repel(aes(lng + 35.35, lat + 35.35, label = paste0("C",ID)), 
                  fontface = 'bold', color = 'red', size = 3,
                  box.padding = unit(0.5, "lines"),
                  point.padding = unit(0.1, "lines"),
                  segment.color = 'black',
                  data = f0[f0$ID < 8,]) +
  theme(axis.line=element_blank(),axis.text.x=element_blank(), axis.text.y=element_blank(),axis.ticks=element_blank(),
        axis.title.x=element_blank(), axis.title.y=element_blank(),legend.position="none",
        panel.background=element_blank(),panel.border=element_blank(),panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),plot.background=element_blank())
plot(g)
pdf('zoom.pdf', width = 4, height = 4)
print(g)
dev.off()
