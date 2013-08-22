# sample2.R

min=1e10
repeat {
	s=sample(1:5)
	first=100*s[1]+10*s[2]+s[3]
	second=10*s[4]+s[5]
	prod=first*second
	if (prod<min) {
		print(paste(first,second,prod))
		min=prod
	}
}



# eof

