#install.packages('LaplacesDemon')
library('LaplacesDemon')

mu = c(0, 0)
x = c(1, 1)
v = 3
mat = 2 * diag(2)
mat.inv = 0.5 * diag(2)
print(dmvt(x, mu, mat,     df=v, log=TRUE))
print(dmvt(x, mu, mat.inv, df=v, log=TRUE))
