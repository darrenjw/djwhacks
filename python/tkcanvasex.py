#!/usr/bin/env python3

from tkinter import *

root = Tk()
c = Canvas(root, width=1000, height=1000)
c.pack()

c.create_rectangle(100, 100, 200, 200, fill="blue", outline = 'black')
c.create_line(100, 300, 300, 300, fill="#990000", width=3)
c.create_oval(100, 400, 150, 450, fill="red", outline = "black")
c.create_oval(200, 400, 201, 401, fill="black", outline="black")
c.create_polygon([300, 400, 400, 400, 350, 500], fill="red", outline="black")

mainloop()
