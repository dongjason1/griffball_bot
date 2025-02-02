from bot import PressKey, ReleaseKey, moveMouse, clickMouse
import numpy as np
import cv2
import math
import time
from win32gui import GetWindowText, GetForegroundWindow
import configparser
from os import path
from win32api import GetSystemMetrics
import mss

'''
loads and returns the config file

if the config file does not exist, create one
'''
def getconfig():
	config = configparser.ConfigParser()
	if path.exists('config.ini'):
		config.read('config.ini')

		setup = config['def']['setup']
		center = eval(config['def']['center'])
		attack_distance = int(config['def']['attack_distance'])
		Starting_time = int(config['def']['Starting_time'])
		map_size = int(float(config['def']["Map_Size"]))
		show = config['def']["show_screen"]
		xlevel = int(config['def']['forward_target_biased'])

		print('Config File Loaded')
		waitsec(Starting_time)

		return setup, center, attack_distance, Starting_time, map_size, show, xlevel
		
	else:
		print('Creating Config File')
		print('Make sure you are MOVING, when the timer ends. (Also avoid looking at yellow things)')
		waitsec(10)

		center = findstart()
		map_size = getMapSize(center)

		# min(center) is a hacky way to get an estimate of the map size
		config['def'] = {'setup': 'True', 'center': str(center), 'attack_distance': '40', 'Starting_time' : '5',"Map_Size" : str(map_size), "show_screen" : "True", 'forward_target_biased' : "10"}
		with open('config.ini', 'w') as configfile:
			config.write(configfile)
		#exit because we're done writing the config file
		raise SystemExit

'''
Finds the center of the motion tracker in halo
'''
def findstart():
	# Take a screenshot, convert to usable in CV2
	sct = mss.mss()
	h = GetSystemMetrics(0)
	w = GetSystemMetrics(1)
	img = sct.grab((0,0,int(h),w))
	img2 = np.asarray(img, dtype='uint8')
	frame = cv2.cvtColor(img2, cv2.COLOR_BGRA2BGR)

	#only keep pixels within this range and make them gray
	color_mask1 = cv2.inRange(frame, np.array([0,190,210]), np.array([50,255,255]))
	yellowmask = cv2.bitwise_and(frame, frame, mask = color_mask1)
	grey = cv2.cvtColor(yellowmask, cv2.COLOR_RGB2GRAY)

	# approximate contours
	contours,_ = cv2.findContours(grey,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

	# get the coordinates of the center, assumes only one contour
	for c in contours:
		obj = cv2.boundingRect(c)
		x,y,w,h = obj
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
		center = (int(x+w/2),int(y+h/2))

	return center

'''
Get motion tracker bounding box

Works because we assume the box is in the bottom left. Max width 
is just our x coordinate and max height is the screen height minus 
our y coordinate
'''
def getMapSize(map_center):
	max_width = map_center[0]
	max_height = GetSystemMetrics(1) - map_center[1]

	return int(min(max_width, max_height))

'''
wait 'sec' seconds
'''
def waitsec(sec):
	for i in range(sec,0,-1):
		print('Starting in: ' + str(i))
		time.sleep(1)

'''
euclidean distance
'''
def calcdistance(x,y,x2=252,y2=318):
	distance = math.sqrt((x-x2)**2 + (y-y2)**2)
	return round(distance,2)

'''
Get the relative angle of two points

x2 and y2 are the center, think of this as the origin (0,0)
x and y are the other point, forming a line to the origin
the angle returned goes from -180 to 180
positive to the right, negative to the left
'''
def calcAngle(x, y, x2, y2):
	if (y-y2)==0:
		return 0
	angle = -(np.arctan((x-x2)/(y-y2)) * 180/3.14159)
	if y2 < y:
		angle = -angle
		if angle >= 0:
			angle = 180 - angle
		else:
			angle = -180 - angle
	return angle

'''
Returns a frame of the screen and a grayscale frame of the red blips
'''
def getframe(leftx=0,topy=0,rightx=GetSystemMetrics(0),bottomy=GetSystemMetrics(1)):
	# Take a screenshot, convert to usable in CV2
	sct = mss.mss()
	im = sct.grab((leftx,topy,rightx,bottomy))
	img2 = np.asarray(im, dtype='uint8')
	frame = cv2.cvtColor(img2, cv2.COLOR_BGRA2BGR)

	#only keep pixels within this range and make them gray
	color_mask2 = cv2.inRange(frame, np.array([0,0,235]), np.array([75,40,255])) #red
	redmask = cv2.bitwise_and(frame, frame, mask = color_mask2)
	grey = cv2.cvtColor(redmask, cv2.COLOR_RGB2GRAY)
	return frame, grey



def main():
	#keyboard scan codes
	W = 0x11
	A = 0x1E
	S = 0x1F
	D = 0x20

	# load config
	setup, center, attack_distance, Starting_time, map_size , show, xlevel = getconfig()
	leftx = int(center[0] - map_size)
	rightx = int(center[0] + map_size)
	topy = int(center[1] + map_size)
	bottomy = int(center[1] - map_size)

	# main loop
	while GetWindowText(GetForegroundWindow()) == "Halo: The Master Chief Collection  " or show == 'True':
		frame, grey = getframe(leftx,bottomy,rightx,topy)
		center = (int(frame.shape[0]/2),int(frame.shape[1]/2))

		if any(frame[center[0]][center[1]]) == 0:
			print('Waiting for game')
			time.sleep(5)

		else:
			contours,_ = cv2.findContours(grey,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

			#Choose the contour closest to the center
			w,h = 0,0 #width and height of a blob
			shortx,shorty,shortw,shorth = 0,0,0,0 #values of the closest blob
			shortest_distance = 5000
			for c in contours:
				x,y,w,h = cv2.boundingRect(c)

				distance = calcdistance(x,y,center[0],center[1])
				if distance < shortest_distance:
					shortest_distance = distance
					shortx,shorty,shortw,shorth = (x,y,w,h)
				else:
					cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
				cv2.putText(frame, '. Distance: ' + str(distance),(x,y), cv2.FONT_HERSHEY_COMPLEX_SMALL,.5,(255, 255, 255))

			# Little extra drawng in our cv2 window
			cv2.rectangle(frame,(shortx,shorty),(shortx+shortw,shorty+shorth),(0,255,255),2)
			cv2.line(frame,(center[0],center[1]+500),(center[0],center[1]-500),(255,0,0))
			cv2.line(frame,(center[0]+500,center[1]),(center[0]-500,center[1]),(255,0,0))
			cv2.line(frame,(center[0]+500,center[1]+xlevel),(center[0]-500,center[1]+xlevel),(0,0,255))

			# angle of the closest dot
			angle = calcAngle(shortx,shorty,center[0], center[1])
				
			# movement and look around commands
			if len(contours) < 1:
				PressKey(W)
				ReleaseKey(A)
				ReleaseKey(D)
				ReleaseKey(S)
			else:
				if shortx+w/2 > center[0]:
					ReleaseKey(A)
					PressKey(D)
				elif shortx+w/2 < center[0]:
					ReleaseKey(D)
					PressKey(A)
				if shorty+h/2 > center[1] + xlevel:
					ReleaseKey(W)
					PressKey(S)
				elif shorty+h/2 < center[1] + xlevel:
					ReleaseKey(S)
					PressKey(W)
				moveMouse(x=2*angle, y=0)

			# strike command
			if shortest_distance < attack_distance and np.absolute(angle) < 10:
				clickMouse()

			# display the cv2 frame
			if show == "True":
				cv2.imshow("Halo_Reach_Griff_Bot",frame)
			# exit situations
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break

if __name__ == "__main__":
	main()