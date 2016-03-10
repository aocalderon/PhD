len <- 24
x = runif(len)
y = x^2 + runif(len, min = -0.1, max = 0.1)

x = seq(0.01,1,0.01)
y = e
plot(x, y)
s <- seq(from = 0.01, to = 1, length = 100)
lines(s, s^2, lty = 2)

df <- data.frame(x, y)
m <- nls(y ~ I(x^power), data = df, start = list(power = 1), trace = T)

m.exp <- nls(y ~ I(a * exp(b * (x-1))), data = df, start = list(a = 0.1, b = 20), trace = T)

plot(x, y)
lines(s, s^2, lty = 2)
lines(s, predict(m, list(x = s)), col = "green")
lines(s, predict(m.exp, list(x = s)), col = "red")
