## tidyverse.R
## Messing about with tidyverse functions

## install.packages("tidyverse")

## https://www.tidyverse.org/
## https://r4ds.had.co.nz/
## https://monashbioinformaticsplatform.github.io/r-more/topics/tidyverse.html
## https://tibble.tidyverse.org/
## https://rstudio.com/resources/cheatsheets/
## https://ggplot2.tidyverse.org/
## https://tutorials.iq.harvard.edu/R/Rgraphics/Rgraphics.html

## dbplyr for querying databases and generating SQL...

library(tidyverse)

tapply(mtcars$mpg, mtcars$cyl, mean)

mtcars %>%
    group_by(cyl) %>%
    summarise(ampg = mean(mpg))

mtcars %>%
    group_by(cyl) %>%
    arrange(desc(mpg))


hist(mtcars$mpg, "FD")
ggplot(mtcars, aes(x = mpg)) + geom_histogram(binwidth=5)

plot(mtcars$hp, mtcars$mpg, pch=19, col=3, cex=0.5, xlab = "Horse power")
ggplot(mtcars, aes(x = hp, y = mpg, color=cyl)) +
    scale_x_continuous(name = "Horse power") +
    geom_point(size=1) +
    geom_smooth() +
    geom_smooth(method = "lm")

ggplot(mtcars, aes(x = hp, y = mpg)) +
    scale_x_continuous(name = "Horse power") +
    geom_point(size=1) +
    facet_wrap(~cyl)



## eof

