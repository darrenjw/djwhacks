#!/usr/bin/env python
# combine.py
# Script to concatenate a bunch of parquet files with identical structure

import pandas as pd
import sys

outfile = "all-results.parquet"

args = sys.argv
if (len(args) < 3):
    print(f'Usage: python {args[0]} f1.parquet f2.parquet ...')
    sys.exit()
files = args[1:]
print(f'Concatenating: {files}')
df = pd.DataFrame()
for f in files:
    print(f)
    dff = pd.read_parquet(f)
    df = pd.concat([df, dff])
print(df)
print(f'Writing: {outfile}')
df.to_parquet(outfile)
print("Done!")


# eof

