#!/usr/bin/env python
# results.py
# Download some results using FastF1

import fastf1
import pandas as pd

start_year = 2014
end_year = 2024
# Fields to extract from the race results data frame:
fields = ['BroadcastName', 'Abbreviation', 'DriverId', 'TeamId', 'Position', 'ClassifiedPosition', 'GridPosition', 'Time', 'Status', 'Points']


def results(year, rnd):
    sess = fastf1.get_session(year, rnd, 'R')
    sess.load()
    res = sess.results
    r, c = res.shape
    if (r == 0):
        print(f'ERROR: failed to download results for {year}, round {rnd}')
        raise 'Yikes!'
    sub = res[fields]
    return(sub)

def full_results(year, rnd):
    res = results(year, rnd)
    res['Year'] = year
    res['Round'] = rnd
    return(res)

df = pd.DataFrame()
for y in range(start_year, end_year+1):
    schedule = fastf1.get_event_schedule(y)
    r, c = schedule.shape
    print(f'{r-1} events in {y}')
    for e in range(1, r):
        res = full_results(y, e)
        d, cc = res.shape
        print(f'Y: {y}, R: {e}, D: {d}')
        df = pd.concat([df, res])

print(df)
df.to_parquet('results.parquet')
print("Done!")


# eof

