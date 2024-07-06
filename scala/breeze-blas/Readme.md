# Using optimised BLAS and LAPACK libraries with Scala Breeze

[Breeze](https://github.com/scalanlp/breeze) is the standard scientific and numerical library for Scala. For linear algebra operations, it builds on top of the Java library, [netlib](https://github.com/luhenry/netlib). This provides a nice interface to BLAS and related libraries which allows the use of native optimised libraries and will also gracefully fall back to using a Java implementations if an optimised native code library can't be found. This is great, but the Java implementations will typically be much slower than optimised native libraries, so if you care about speed it is important to install optimised libraries on your system and configure netlib to use them.

See the [netlib Readme](https://github.com/luhenry/netlib/blob/master/README.md) for details of installing blas libraries and setting the relevant system properties. Briefly, you can override default settings by setting the properties `blas`, `lapack` and `arpack`. Each of this can be set by using either `nativeLib`, to specify the name of a library in your system library search path or `nativeLibPath`, to set the full path to the library you require. Full examples of the two approaches are:
```
-Ddev.ludovic.netlib.blas.nativeLib=libopenblas.so

-Ddev.ludovic.netlib.blas.nativeLibPath=/usr/lib/x86_64-linux-gnu/libopenblas.so
```
Obviously these need to be customised to your requirement. `lapack` and `arpack` properties are set similarly.

What the netlib readme doesn't discuss is how to set these properties in Scala projects, or how to check/verify the libraries being used in Scala Breeze projects. These are discussed below.

## Setting netlib properties in Scala projects


## Verifying netlib instances in Scala Breeze projects


