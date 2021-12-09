
from tkinter import *

root = Tk()
c = Canvas(root, width=1000, height=1000)
c.pack()


c.create_rectangle(100, 100, 200, 200, fill="blue", outline = 'black')
