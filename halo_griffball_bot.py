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

# Helper Functions
def waitsec(sec):
	for i in range(sec,0,-1):
		print('Starting in: ' + str(i))
		time.sleep(1)

def calcdistance(x,y,x2=252,y2=318):
	distance = math.sqrt((x-x2)**2 + (y-y2)**2)
	return round(distance,2)

def calcAngle(x, y, x2, y2):
	if (y-y2)==0:
		return 0
	angle = np.arctan((x-x2)/(y-y2)) * 180/3.14159
	return -angle

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
		center = (x+w/2,y+h/2)

	leftx, rightx, topy, bottomy = checkbounds(center,min(center))

	return center

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

		config['def'] = {'setup': 'True', 'center': str(center), 'attack_distance': '40', 'Starting_time' : '5',"Map_Size" : str(min(center)), "show_screen" : "True", 'forward_target_biased' : "10"}
		with open('config.ini', 'w') as configfile:
			config.write(configfile)

def checkbounds(center,lowest_num):
	bbx = (int(center[0] - lowest_num), int(center[0] + lowest_num))
	bby = (int(center[1] + lowest_num), int(center[1] - lowest_num))
	
	for i in bbx:
		if i > GetSystemMetrics(0):
			sub = i - GetSystemMetrics(0)
			lowest_num = lowest_num - sub
	for i in bby:
		if i > GetSystemMetrics(1):
			sub = i - GetSystemMetrics(1)
			lowest_num = lowest_num - sub

	leftx = int(center[0] - lowest_num)
	topy = int(center[1] + lowest_num)
	rightx = int(center[0] + lowest_num)
	bottomy = int(center[1] - lowest_num)

	return leftx, topy, rightx, bottomy




W = 0x11
A = 0x1E
S = 0x1F
D = 0x20

setup,center, attack_distance, Starting_time, map_size , show, xlevel = getconfig()
lowest_num = map_size
leftx, topy, rightx, bottomy = checkbounds(center,lowest_num)

frame, grey = getframe(leftx,bottomy,rightx,topy)

center = (int(frame.shape[0]/2),int(frame.shape[1]/2))
while GetWindowText(GetForegroundWindow()) == "Halo: The Master Chief Collection  " or show == 'True':
	skip = False
	frame, grey = getframe(leftx,bottomy,rightx,topy)

	if any(frame[center[0]][center[1]]) == 0:
		skip = True
		print('Waiting for game')
		time.sleep(5)

	if not skip:

		contours,_ = cv2.findContours(grey,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

		w,h = 0,0
		shortx,shorty,shortw,shorth = 0,0,0,0
		index = 0
		shortest_distance = 5000
		for c in contours:
			index = index + 1
			obj = cv2.boundingRect(c)
			x,y,w,h = obj

			distance = calcdistance(x,y,center[0],center[1])
			if distance < shortest_distance:
				shortest_distance = distance
				shortx,shorty,shortw,shorth = obj
			else:
				cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
			cv2.putText(frame, str(index) + '. Distance: ' + str(distance),(x,y), cv2.FONT_HERSHEY_COMPLEX_SMALL,.5,(255, 255, 255))
		cv2.rectangle(frame,(shortx,shorty),(shortx+shortw,shorty+shorth),(0,255,255),2)
		cv2.line(frame,(center[0],center[1]+500),(center[0],center[1]-500),(255,0,0))
		cv2.line(frame,(center[0]+500,center[1]),(center[0]-500,center[1]),(255,0,0))
		cv2.line(frame,(center[0]+500,center[1]+xlevel),(center[0]-500,center[1]+xlevel),(0,0,255))

		angle = calcAngle(shortx,shorty,center[0], center[1])
			
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
				angle = -angle
				if angle >= 0:
					angle = 180 - angle
				else:
					angle = -180 - angle
				ReleaseKey(W)
				PressKey(S)
			elif shorty+h/2 < center[1] + xlevel:
				ReleaseKey(S)
				PressKey(W)
			moveMouse(x=2*angle, y=0)
		try:
			if shortest_distance < attack_distance and np.absolute(angle) < 10:
				clickMouse()
		except:
			print('First-Time Startup: Complete')
			print('[!] Restart Script [!]')
			sleep(3)

		if show == "True":
			cv2.imshow("Halo_Reach_Griff_Bot",frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

cv2.destroyAllWindows()
print('[!] Done, Closing Window')

