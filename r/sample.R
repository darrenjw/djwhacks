# sample.R

target=66.99

repeat {
	s=sample(0:9,8)
	num=10*s[1]+s[2]+0.1*s[3]+0.01*s[4]
	den=10*s[5]+s[6]+0.1*s[7]+0.01*s[8]
	diff=num-den
	err=abs(diff-target)
	if (err<0.001)
		print(paste(num,den,diff))
}



# eof

