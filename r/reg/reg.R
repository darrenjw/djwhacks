# reg.R

df=read.csv("test.csv")
mf=model.frame(log(salary)~age+gender,data=df)
mr=model.response(mf)
mm=model.matrix(log(salary)~age+gender,data=mf)
lmf=lm.fit(mm,mr)
lmf$coefficients



# eof


