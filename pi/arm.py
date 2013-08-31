# arm.py
# ROBOT ARM CONTROL PROGRAM

# import the USB and Time libraries into Python
import usb.core, usb.util, time, sys

# Allocate the name 'RoboArm' to the USB device
RoboArm = usb.core.find(idVendor=0x1267, idProduct=0x0000)

# Check if the arm is detected and warn if not
if RoboArm is None:
	raise ValueError("Arm not found")

# Create a variable for duration
Duration=0.4

# Define a procedure to execute each movement
def MoveArm(Duration, ArmCmd):
	# Start the movement
	RoboArm.ctrl_transfer(0x40,6,0x100,0,ArmCmd,1000)
	# Stop the movement after waiting specified duration
	time.sleep(Duration)
	ArmCmd=[0,0,0]
	RoboArm.ctrl_transfer(0x40,6,0x100,0,ArmCmd,1000)

# Give the arm some commands
#MoveArm(Duration,[0,1,0]) # Rotate Base anti-Clockwise
#MoveArm(Duration,[0,2,0]) # Rotate Base Clockwise
#MoveArm(Duration,[64,0,0]) # Shoulder Up
#MoveArm(Duration,[128,0,0]) # Shoulder Down
#MoveArm(Duration,[16,0,0]) # Elbow Up
#MoveArm(Duration,[32,0,0]) # Elbow Down
#MoveArm(Duration,[4,0,0]) # Wrist Up
#MoveArm(Duration,[8,0,0]) # Wrist Down
#MoveArm(Duration,[2,0,0]) # Grip Open
#MoveArm(Duration,[1,0,0]) # Grip Close
#MoveArm(Duration,[0,0,1]) # Light On
#MoveArm(Duration,[0,0,0]) # Light Off

# Now make it properly programmable...

def move(string):
	for char in string:
		if char=='q':
			MoveArm(Duration,[2,0,0]) # Grip Open
		elif char=='a':
			MoveArm(Duration,[1,0,0]) # Grip Close
		elif char=='w':
			MoveArm(Duration,[4,0,0]) # Wrist Up
		elif char=='s':
			MoveArm(Duration,[8,0,0]) # Wrist Down
		elif char=='e':
			MoveArm(Duration,[16,0,0]) # Elbow Up
		elif char=='d':
			MoveArm(Duration,[32,0,0]) # Elbow Down
		elif char=='r':
			MoveArm(Duration,[64,0,0]) # Shoulder Up
		elif char=='f':
			MoveArm(Duration,[128,0,0]) # Shoulder Down
		elif char=='t':
			MoveArm(Duration,[0,1,0]) # Rotate Base anti-Clockwise
		elif char=='g':
			MoveArm(Duration,[0,2,0]) # Rotate Base Clockwise
		else:
			MoveArm(Duration,[0,0,0]) # all off (pause)


if __name__=='__main__':
	# move("qawsedrftg")
	print "Enter a string of moves..."
	while (1==1):
		s=sys.stdin.readline()
		move(s)
		


# eof



