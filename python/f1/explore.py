#!/usr/bin/env python
# explore.py
# Download some results using FastF1
# Figure out how it all works...

import fastf1

# Have a bit of a muck around...
year = 2024
schedule = fastf1.get_event_schedule(year)
r, c = schedule.shape
print(r)
print(schedule.columns)
print(schedule.get('Country'))

r12 = schedule.get_event_by_round(12)
print(r12['EventName'])
sess = fastf1.get_session(year, 12, 'R') # get the race
print(sess)
sess.load() # need to load the session before accessing the results
res = sess.results
print(res)
print(res.shape)
print(res.columns)
print(res['Abbreviation'])
print(res['TeamId'])
print(res['ClassifiedPosition'])
print(res['Points'])
print(res.loc['44'])
fields = ['BroadcastName', 'Abbreviation', 'DriverId', 'TeamId', 'Position', 'ClassifiedPosition', 'GridPosition', 'Time', 'Status', 'Points']
print(res[fields])

# All just pandas foo from here on in...

def results(year, rnd):
    sess = fastf1.get_session(year, rnd, 'R')
    sess.load()
    res = sess.results
    sub = res[fields]
    return(sub)

res = results(year, 12)
print(res)

def full_results(year, rnd):
    res = results(year, rnd)
    res['Year'] = year
    res['Round'] = rnd
    return(res)

res = full_results(year, 12)
print(res)



# eof

