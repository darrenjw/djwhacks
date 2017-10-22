# balls.py
# This python script requires "pygame" to be installed


import pygame
import math
import  random
background_colour = (255,255,255)
(width, height) = (800, 800)
W=40
BIGW=W*1.2
NUMBALLS=10
radius=400
CX=width/2
CY=height/2
v0=0.5
v1=0.1
sep=16
f=0.1/sep
def factor(x):

    if x>sep:
        return 0
    else:
        return x*(x-sep)*f
    if x<0:
        return 0.5
    if x<16:
        return -0.1
    return 0.0

def AddBall(x,y,s):
    b=Ball()
    b.x,b.y=x,y
    b.size=s
    balls.append(b)
    return b
def RemoveBall(b):
    global balls
    balls.remove(b)

def grow(s):
    return (1.0 +random.randint(0,20)/8000.0*BIGW/s)

class Ball:
    def __init__(self):
        self.x = CX+random.randint(0,radius)-radius/2
        self.y = CY+random.randint(0,radius)-radius/2
        self.size = W+random.randint(0,	20)-10
        self.colour = (0, 0, 255)
        self.thickness = 4

    def display(self):
        pygame.draw.circle(screen, self.colour, (int(self.x), int(self.y)), int(self.size),self.thickness)

    def getdelta(self,balls):
        dx=dy=0
        for b in balls:
            if b!=self:
                dx0,dy0,d0=b.dist(self)
                delta=d0-self.size-b.size
                v=factor(delta)               
                dx+=-dx0/d0*v
                dy+=-dy0/d0*v
 
            
			
        return dx,dy	
    def evolve(self):
        if random.random()<0.001:
            RemoveBall(self)
            return
        self.size*=grow(self.size)
        if self.size>BIGW:
            self.size/=math.sqrt(2.0)
            dx,dy=random.normalvariate(0,10),random.normalvariate(0,10)
            print "split",dx,dy
            b=AddBall(self.x,self.y,self.size*grow(self.size))
            self.move(dx,dy)
            b.move(-dx,-dy)
	
    def move(self,dx,dy):
        self.x+=dx
        self.y+=dy

    def dist(self,other):
        dx=self.x-other.x
        dy=self.y-other.y
        return dx,dy,math.sqrt( dx*dx+dy*dy)

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Tutorial 2')
screen.fill(background_colour)
clock = pygame.time.Clock()


balls=[Ball() for x in range(NUMBALLS)]

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False


    screen.fill(background_colour)


    
    for i,ball in enumerate(balls):
        ball.evolve()
   

    deltas=[ ball.getdelta(balls) for ball in balls]


    for i,ball in enumerate(balls):
   	    dx,dy=deltas[i]
	    ball.move(dx,dy)

    for ball in balls:
	    ball.display()
 
    pygame.display.flip()


    #cap the framerate
    clock.tick(40)
