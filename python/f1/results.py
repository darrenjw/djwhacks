#!/usr/bin/env python
# results.py
# Download some results using FastF1

import fastf1

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

# All just pandas foo from here on in...



# eof

