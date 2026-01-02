# Maps in R

First register for an API key with Stadia Maps

<https://client.stadiamaps.com/signup/>

then check API key at:

<https://client.stadiamaps.com/dashboard/overview/>

Register it with `ggmap` using:

```R
register_stadiamaps(key="MY API KEY", write=TRUE)
```

See `?register_stadiamaps` in the `ggmap` package for more info.

Note that they have a basic free account for personal use that does not require any credit card details.


