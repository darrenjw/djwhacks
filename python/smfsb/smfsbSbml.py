#!/usr/bin/env python3
# smfsbSbml.py
# Import SBML models into an smfsb Spn

import libsbml as sb
import smfsb
import sys

def sbml2spn(filename, verb=False):
    d = sb.readSBML(filename)
    m = d.getModel()
    if (m == None):
        print("Can't parse SBML file: "+filename)
        sys.exit(1)
    print("Success")


    

# Test code

if (__name__ == '__main__'):
    sbml2spn("lambda.xml")
    

# eof


