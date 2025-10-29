import numpy as np
import dlib
from imutils import face_utils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

sleep = 0
drowsy = 0
active = 0

def compute(ptA,ptB):
	dist = np.linalg.norm(ptA - ptB)
	return dist

def blinked(a,b,c,d,e,f):
	up = compute(b,d) + compute(c,e)
	down = compute(a,f)
	ratio = up/(2.0*down)

	if(ratio>0.25):
		return 2
	elif(ratio>0.21 and ratio<=0.25):
		return 1
	else:
		return 0

""" Returns Status which consists of four states (1-SLEEPING, 2-DROWSY, 3-ACTIVE, 4-UNKNOWN) """
def getPrediction(frame):
	global sleep,drowsy,active
	faces = detector(frame)
	l = []
	for face in faces:
		landmarks = predictor(frame, face)
		landmarks = face_utils.shape_to_np(landmarks)

		left_blink = blinked(landmarks[36],landmarks[37], 
			landmarks[38], landmarks[41], landmarks[40], landmarks[39])
		right_blink = blinked(landmarks[42],landmarks[43], 
			landmarks[44], landmarks[47], landmarks[46], landmarks[45])
		
		if(left_blink==0 or right_blink==0):
			sleep+=1
			drowsy=0
			active=0
			if(sleep>6):
				l.append(1)

		elif(left_blink==1 or right_blink==1):
			sleep=0
			active=0
			drowsy+=1
			if(drowsy>6):
				l.append(2)

		else:
			drowsy=0
			sleep=0
			active+=1
			if(active>6):
				l.append(3)
	return 4 if not l else max(l)