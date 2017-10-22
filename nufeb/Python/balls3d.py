# balls3d.py
# This python script requires "panda3d" to be installed

from direct.showbase.ShowBase import ShowBase
#from direct.directbase import DirectStart
from panda3d.ode import OdeWorld, OdeBody, OdeMass
from panda3d.core import Quat
from panda3d.core import *


import math
import  random
background_colour = (255,255,255)
(width, height) = (800, 800)
W=40
BIGW=W*1.2
NUMBALLS=100
radius=40
CX=width/2
CY=height/2
CZ=0
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

def AddBall(x,y,z,s,app):
    b=Ball(app)
    b.x,b.y,b.z=x,y,z
    b.size=s
    app.balls.append(b)
    return b


def grow(s):
    return (1.0 +random.randint(0,20)/8000.0*BIGW/s)

class Ball:
    def __init__(self,app):
        self.app=app
        self.x = CX+random.randint(0,radius)-radius/2
        self.y = CY+random.randint(0,radius)-radius/2
        self.z = CZ+random.randint(0,radius)-radius/2
        self.size = W+random.randint(0,	20)-10
        self.colour = (0, 0, 255)
        self.thickness = 4

       # Load the smiley model which will act as our iron ball
        if 0:#random.random()<0.5:
            self.sphere = app.loader.loadModel("misc/sphere.egg")
        else:
            self.sphere = app.loader.loadModel("smiley.egg")
        #self.sphere=sphere(size = self.size, position = P3(self.x,self.y,self.z))
        self.sphere.reparentTo(app.render)
        self.sphere.setPos(self.x,self.y,self.z)
        self.sphere.setScale(self.size)
        self.sphere.setTransparency(TransparencyAttrib.MAlpha)
        self.sphere.setColor(0.7, 0.4, 0.4,0.4)


    def getdelta(self,balls):
        dx=dy=dz=0
        for b in balls:
            if b!=self:
                dx0,dy0,dz0,d0=b.dist(self)
                delta=d0-self.size-b.size
                v=factor(delta)               
                dx+=-dx0/d0*v
                dy+=-dy0/d0*v
                dz+=-dz0/d0*v
            
			
        return dx,dy,dz
    def evolve(self):
        if random.random()<0.001:
            self.app.balls.remove(self)
            
            return
        self.size*=grow(self.size)
        if self.size>BIGW:
            self.size/=math.pow(2.0,1.0/3)
            dx,dy,dz=random.normalvariate(0,10),random.normalvariate(0,10),random.normalvariate(0,10)
            b=AddBall(self.x,self.y,self.z,self.size*grow(self.size),self.app)
            self.move(dx,dy,dz)
            b.move(-dx,-dy,-dz)

	
    def move(self,dx,dy,dz):
        self.x+=dx
        self.y+=dy
        self.z+=dz
    def update(self):
        self.sphere.setPos(self.x,self.y,self.z)
        self.sphere.setScale(self.size)     

    def dist(self,other):
        dx=self.x-other.x
        dy=self.y-other.y
        dz=self.z-other.z
        return dx,dy,dz,math.sqrt( dx*dx+dy*dy+dz*dz)



class MyApp(ShowBase):
 
    def __init__(self):
        ShowBase.__init__(self)
   
        dlight = DirectionalLight('dlight')
        dlight.setColor(VBase4(0.8, 0.8, 0.5, 1))
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(0, -60, 0)
        self.render.setLight(dlnp)

        self.balls=[Ball(self) for x in range(NUMBALLS)]
 
         
        
        # Setup our physics world and the body
        self.world = OdeWorld()
        self.world.setGravity(0, 0, -9.81)
        if 0:
            self.body = OdeBody(self.world)
            M = OdeMass()
            M.setSphere(7874, 1.0)
            self.body.setMass(M)
            self.body.setPosition(self.sphere.getPos(self.render))
            self.body.setQuaternion(self.sphere.getQuat(self.render))
         
        ## Set the camera position
        self.disableMouse()
        self.angle=0.0
        self.camera.setPos(1000,1000,1000)
        self.camera.lookAt(0, 0, 0)
        #self.enableMouse()
        # Create an accumulator to track the time since the sim
        # has been running
        self.deltaTimeAccumulator = 0.0
        # This stepSize makes the simulation run at 60 frames per second
        self.stepSize = 1.0 / 60.0
         
    
         
        taskMgr.doMethodLater(1.0, self.simulationTask, "Physics Simulation")
 
   


    def simulationTask(self,task):
        if 0:
            # Add the deltaTime for the task to the accumulator
            self.deltaTimeAccumulator += globalClock.getDt()
            while self.deltaTimeAccumulator > self.stepSize:
                # Remove a stepSize from the accumulator until
                 # the accumulated time is less than the stepsize
                self.deltaTimeAccumulator -= self.stepSize
                # Step the simulation
                self.world.quickStep(self.stepSize)
        for ball in self.balls:
            ball.evolve()  
        deltas=[ ball.getdelta(self.balls) for ball in self.balls]
        for i,ball in enumerate(self.balls):
            dx,dy,dz=deltas[i]
            ball.move(dx,dy,dz)
            ball.update()
        #print "moved"
        self.camera.setPos(CX+2500*math.sin(self.angle),CY+2500*math.cos(self.angle),CZ)
        self.camera.lookAt(CX,CY,CZ)
        self.angle+=0.01
        return task.cont 



app = MyApp()
app.run()
