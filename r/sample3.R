# sample2.R

target=10000
best=1000
repeat {
	s=sample(1:5)
	first=100*s[1]+10*s[2]+s[3]
	second=10*s[4]+s[5]
	prod=first*second
	err=abs(prod-target)
	if (err<best) {
		print(paste(first,second,prod,err))
		best=err
	}
}



# eof

