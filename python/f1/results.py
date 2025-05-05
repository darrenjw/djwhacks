#!/usr/bin/env python
# results.py
# Download some results using FastF1

import fastf1
import pandas as pd
import sys

# Fields to extract from the race results data frame:
fields = ['BroadcastName', 'Abbreviation', 'DriverId', 'TeamId', 'Position', 'ClassifiedPosition', 'GridPosition', 'Time', 'Status', 'Points']


# results for a given round in a given season
def results(year, rnd):
    sess = fastf1.get_session(year, rnd, 'R')
    sess.load()
    res = sess.results
    r, c = res.shape
    if (r == 0):
        print(f'ERROR: failed to download results for {year}, round {rnd}')
        raise
    sub = res[fields]
    return(sub)

# just retries "results" a few times in case of API issues
def results_retry(year, rnd):
    tries = 5
    for i in range(tries):
        try:
            return(results(year, rnd))
        except:
            if i < tries - 1:
                continue
            else:
                print(f'ERROR: failed {tries} times - giving up!')
                raise
        break

# "results", but with year and round as additional columns
def full_results(year, rnd):
    res = results_retry(year, rnd)
    res['Year'] = year
    res['Round'] = rnd
    return(res)

# all "results" for a given season
def season_results(year):
    df = pd.DataFrame()
    schedule = fastf1.get_event_schedule(year)
    r, c = schedule.shape
    print(f'{r-1} events in {year}')
    rounds = schedule['RoundNumber'] # list of rounds in the season
    rounds = rounds[rounds > 0] # drop phantom/test events
    for e in rounds:
        res = full_results(year, e)
        d, cc = res.shape
        print(f'Y: {year}, R: {e}, D: {d}')
        df = pd.concat([df, res])
    return(df)




# Code to run if run as a script

if __name__ == '__main__':
    args = sys.argv
    if (len(args) != 2):
        print(f"Usage: python {args[0]} <year>")
        sys.exit()
    year = args[1]
    df = season_results(int(year))
    print(df)
    resFile = f'results-{year}.parquet'
    df.to_parquet(resFile)
    print(f'Results written to {resFile}')
    print("Done!")


# eof

