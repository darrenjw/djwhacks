# Based on snakebit.py from:
# https://github.com/martinohanlon/microbit-micropython
# and just hacked around a bit to figure out what is going on
# Modifications by @andrewjp


import microbit
import math
import random
class SnakeBit():
    UP = 0
    DOWN = 1
    RIGHT = 2
    LEFT = 3
    
    SNAKEBRIGHTNESS = 9
    APPLEBRIGHTNESS = 5
    SAMPLETIME = 50
    SAMPLESPERMOVE = 10
    
    def __init__(self):
        pass
    
    def startGame(self):
        microbit.display.clear()
        self.direction = self.UP
        self.length = 2
        self.tail = []
        self.tail.insert(0, [2, 4])
        self.apples=[]
        
        self.eaten=0
        self.createApples()
        self.score = 0
        
        self.bstates0=self.getbuttons()
        
        
        
        playing = True
        
        samples = 0
        self.ctr=0
        while(playing):
            #keep looping around, if the button is pressed, move the snake immediately, 
            #otherwise move it when the sample time is reached
            self.ctr+=1
            self.drawApples()
            microbit.sleep(self.SAMPLETIME)
            self.updatebuttons()
            moved=0
            if self.bchange[0]==1:
                moved=1
                self.left()
            elif self.bchange[1]==1:
                moved=1
                self.right()
                
            samples = samples + 1
            if moved or samples == self.SAMPLESPERMOVE:
                playing = self.move()
                samples = 0
                
        microbit.display.scroll("Score = " + str(self.score), 100)
        microbit.display.clear()

    def getbuttons(self):
        return [microbit.button_a.is_pressed(),microbit.button_b.is_pressed()]
    def updatebuttons(self):
        self.bstates=self.getbuttons()
        self.bchange=[ self.bstates[i]-self.bstates0[i] for i in [0,1]]
        self.bstates0=self.bstates[:]
        
  

    def createApple(self):
        badApple = True
        #try and fnd a location for the apple
        while(badApple):
            x = random.randint(0,4)
            y = random.randint(0,4)
            badApple = self.checkCollision(x, y)
        return  [x, y]
        
    def createApples(self):
        self.napples=1#+int(self.eaten/3)
        for i in range(self.napples-len(self.apples)):
            self.apples.append(self.createApple())
            
    def drawApples(self):
        for x,y in self.apples:
            microbit.display.set_pixel(x, y, [4,7,9,9,7,5,4,4,4,4][self.ctr*self.napples %10])

    def checkCollision(self, x, y):
        #is this outside the screen
        if x > 4 or x < 0 or y > 4 or y < 0:
            return True
        else:
            #or in the snakes tail
            for segment in self.tail:
                if segment[0] == x and segment[1] == y:
                    return True
            else:
                return False

    def addSegment(self, x, y):
        #create the new segment of the snake
        #microbit.display.set_pixel(x, y, self.SNAKEBRIGHTNESS)
        self.tail.insert(0, [x, y])
        
        #do I need to clear a segment
        if len(self.tail) > self.length:
            lastSegment = self.tail[-1]
            microbit.display.set_pixel(lastSegment[0], lastSegment[1], 0)
            self.tail.pop()
            
        for i,(x,y) in enumerate(self.tail):
            b=9-int(i/2)
            if b<2:
                b=2
            microbit.display.set_pixel(x,y,b)
            

    def move(self):
        #work out where the new segment of the snake will be
        newSegment = [self.tail[0][0], self.tail[0][1]]
        if self.direction == self.UP:
            newSegment[1] -= 1
        elif self.direction == self.DOWN:
            newSegment[1] += 1
        elif self.direction == self.LEFT:
            newSegment[0] -= 1
        elif self.direction == self.RIGHT:
            newSegment[0] += 1

        if self.checkCollision(newSegment[0], newSegment[1]):
            #game over
            snakehead = self.tail[0]
            for flashHead in range(0,5):
                microbit.display.set_pixel(snakehead[0], snakehead[1], self.SNAKEBRIGHTNESS)
                microbit.sleep(200)
                microbit.display.set_pixel(snakehead[0], snakehead[1], 0)
                microbit.sleep(200)
            
            return False
            
        else:
            self.addSegment(newSegment[0], newSegment[1])

            #has the snake eaten the apple?
            for i,(x,y) in enumerate(self.apples):
                if newSegment[0] == x and newSegment[1] == y:
                    self.length += 1
                    self.score += 10
                    self.apples=self.apples[:i]+self.apples[i+1:]
                    self.eaten+=1
                    
            self.createApples()

            return True

    def left(self):
        if self.direction == self.RIGHT:
            self.direction = self.UP
        elif self.direction == self.UP:
            self.direction = self.LEFT
        elif self.direction == self.LEFT:
            self.direction = self.DOWN
        elif self.direction == self.DOWN:
            self.direction = self.RIGHT

    def right(self):
        if self.direction == self.RIGHT:
            self.direction = self.DOWN
        elif self.direction == self.DOWN:
            self.direction = self.LEFT
        elif self.direction == self.LEFT:
            self.direction = self.UP
        elif self.direction == self.UP:
            self.direction = self.RIGHT

snake = SnakeBit()
snake.startGame()



