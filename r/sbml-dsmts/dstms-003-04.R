# dstms-003-04.R
# Mathematical analysis of 003-04
# DJW, 19/8/15

library(Matrix)
k1=0.001
k2=0.01
Q=matrix(0,31,31)
Q[row(Q)==col(Q)+1]=k2*(1:30)
Q[row(Q)==col(Q)-1]=0.5*k1*(100-2*(0:29))*(99-2*(0:29))
Q[31,1]=0.5*k1*(100-2*30)*(99-2*30)
diag(Q)=-apply(Q,1,sum)
print(Q)

pi0=rep(0,31)
pi0[1]=1
m=rep(0,50)
s=rep(0,50)
P1=expm(Q)
print(P1)
for (t in 1:50) {
  pi0 = pi0 %*% P1
  print(pi0)
  m[t]=sum((0:30)*pi0)
  s[t]=sqrt(sum((0:30)*(0:30)*pi0)-m[t]*m[t])
}
m=ts(c(0,m),start=0)
s=ts(c(0,s),start=0)
print("Mean for P2:")
print(m)
print("SD for P2:")
print(s)
mp=100-2*m
sp=2*s

op=par(mfrow=c(1,2))
plot(mp,ylim=c(0,100))
lines(m,lty=2)
plot(sp)
lines(s,lty=2)
par(op)

df=data.frame(t=0:50,mp=mp,sp=sp,mp2=m,sp2=s)
print(df)
write.csv(df,"d003-04.csv")


# eof


