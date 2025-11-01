## image for stats group page...

bvn = function(x, y)
    dnorm(x, 0, 1)*dnorm(y, 0, 2.5)

coords = expand.grid(seq(-5, 5, len=100), seq(-5, 5, len=100))
z = bvn(coords[,1], coords[,2])
zm = matrix(z, nrow=100)

pdf("picture.pdf", 12, 12)
persp(zm, axes=FALSE, box=FALSE, col=4, theta=50)
dev.off()

## eof

