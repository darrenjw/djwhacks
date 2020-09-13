# SimpleGame.py

places = {}

places['start'] = {}
places['start']['desc'] = 'You are in a dark forest at a fork in the road'
places['start']['choices'] = ['left', 'right']
places['start']['dest'] = ['lake', 'cave']

places['lake'] = {}
places['lake']['desc'] = 'You are next to a misty lake. You could go in to the lake or take the path down the valley.'
places['lake']['choices'] = ['in', 'down']
places['lake']['dest'] = ['dead', 'cave']

places['cave'] = {}
places['cave']['desc'] = 'You are at a spooky cave. You can go in, or head back to the forest.'
places['cave']['choices'] = ['in', 'back']
places['cave']['dest'] = ['dead', 'start']

places['dead'] = {}
places['dead']['desc'] = 'You are dead.'
places['dead']['choices'] = []
places['dead']['dest'] = []

# start location:
loc = 'start'

while True:
    print(places[loc]['desc'])
    i = 0
    for choice in places[loc]['choices']:
        print(i, choice)
        i = i+1
    s = input()
    j = int(s)
    if ((j >= 0) & (j < len(places[loc]['choices']))):
        loc = places[loc]['dest'][j]
        



# eof
