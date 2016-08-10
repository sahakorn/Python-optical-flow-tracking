import numpy as np
import cv2
import time

help_message = '''
USAGE: optical_flow.py [<video_source>]
Keys:
 1 - toggle HSV flow visualization
 2 - toggle glitch
'''
count = 0
def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:      
	cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx+fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

if __name__ == '__main__':
    import sys
    print help_message
    try: fn = sys.argv[1]
    except: fn = 0

    cam = cv2.VideoCapture(fn)
    ret, prev = cam.read()
   
    prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    show_hsv = True
    show_glitch = False
    cur_glitch = prev.copy()
	
    while True:
        ret, img = cam.read()
	vis = img.copy()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	flow = cv2.calcOpticalFlowFarneback(prevgray, gray, 0.5, 5, 15, 3, 5, 1.1, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
	prevgray = gray
	cv2.imshow('flow', draw_flow(gray, flow))
	if show_hsv:
	    gray1 = cv2.cvtColor(draw_hsv(flow), cv2.COLOR_BGR2GRAY)
	    thresh = cv2.threshold(gray1, 25, 255, cv2.THRESH_BINARY)[1]
	    thresh = cv2.dilate(thresh, None, iterations=2)
	    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 
		# loop over the contours
	    for c in cnts:
		# if the contour is too small, ignore it
	    	(x, y, w, h) = cv2.boundingRect(c)
		if w > 100 and h > 100 and w < 900 and h < 680:
			cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 4)
			cv2.putText(vis,str(time.time()), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),1)
	   		
	    cv2.imshow('Image', vis)
	if show_glitch:
	    cur_glitch = warp_flow(cur_glitch, flow)
	    cv2.imshow('glitch', cur_glitch)
	ch = 0xFF & cv2.waitKey(5)
	if ch == 27:
	    break
	if ch == ord('1'):
	    show_hsv = not show_hsv
	    print 'HSV flow visualization is', ['off', 'on'][show_hsv]
	if ch == ord('2'):
	    show_glitch = not show_glitch
	    if show_glitch:
	        cur_glitch = img.copy()
	    print 'glitch is', ['off', 'on'][show_glitch]

    cv2.destroyAllWindows() 		
