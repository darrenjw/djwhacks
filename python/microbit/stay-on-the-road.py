import microbit
import math
import random



class Scrambler():
    
    SAMPLETIME = 50
    
    
    def __init__(self):
        w=self.w=6
        self.board=[[0,w],[0,w],[0,w],[0,w],[0,w]]
        self.samplespermove=10
    
    def startGame(self):
        microbit.display.clear()
        self.ox=0
        
       
        self.score = 0
        
        self.bstates0=self.getbuttons()
        
        
        
        playing = True
        
        samples = 0
        self.ctr=0
        self.move=0
        while(playing):
            #keep looping around, if the button is pressed, move the snake immediately, 
            #otherwise move it when the sample time is reached
            self.ctr+=1
            
            
            
            microbit.sleep(self.SAMPLETIME)
            self.updatebuttons()
            moved=0
            if self.bchange[0]==1:
                moved=1
                self.ox+=1
            elif self.bchange[1]==1:
                moved=1
                self.ox-=1

            samples = samples + 1
            if moved or samples >= self.samplespermove:
                self.move+=1
                if self.move%40 ==0:
                    self.samplespermove-=1
                if self.move %50 ==0:
                    self.w-=1
                if self.w<2:
                    self.w=2
                self.score+=10
                self.drawBoard()
                
                #check collision
                left,right=self.board[-1]
                x=2-self.ox
                if x<=left or x>=right:
                    break   
                self.moveBoard()
                samples = 0
            self.drawPlayer()
            #if self.ox<0 and self.ctr>100:
               # break
        #microbit.display.scroll(str(self.ox))
        
        microbit.display.scroll("Score = " + str(self.score), 100)
        microbit.display.clear()

    def moveBoard(self):
        left,right=self.board[0]
        direc=random.randint(0,2)-1
        left+=direc
        right+=direc
        if right-left>self.w:
            left+=(right-left-self.w)
        
        self.board=[[left,right]]+self.board[:4]
    def getbuttons(self):
        return [microbit.button_a.is_pressed(),microbit.button_b.is_pressed()]
    def updatebuttons(self):
        self.bstates=self.getbuttons()
        self.bchange=[ self.bstates[i]-self.bstates0[i] for i in [0,1]]
        self.bstates0=self.bstates[:]
        
  

   
            
    def drawBoard(self):
        for row,(left,right) in enumerate(self.board):
            left+=self.ox
            right+=self.ox            
            for col in range(5):
 
                b=0
                if col<=left:
                    b=8-(left-col)*2
                elif col>=right:
                    b=8-(col-right)*2
                if b<0:
                    b=0
              
                microbit.display.set_pixel(col,row,b)
    def drawPlayer(self):
        b=[4,7,9,9,7,5,4,4,4,4][(self.ctr*2) %10]
        microbit.display.set_pixel(2,4,b)
   
    
            

    

    
game = Scrambler()
game.startGame()

