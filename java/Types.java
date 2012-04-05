/*
Types.java
Class to illustrate the numeric type problem in Java

*/

public class Types<T extends Number>
{

    public T myAdd(T a,T b)
    {
	return(a+b);
    }

}

/* eof */

