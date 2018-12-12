import numpy as np
import cv2

#cap = cv2.VideoCapture('./mp4_movies/ttc.mp4')
cap = cv2.VideoCapture(0)


# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.000001,
                       minDistance = 1,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
foe = None
framecount = 0
while(1):
    if framecount < 5:
        framecount += 1
        continue
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    print(len(p0))
    diff = p1-p0
    a2 = [(x**2,x*y,y**2,(i*x+j*y)*x,(i*x+j*y)*y) for ([[i,j]],[[x,y]]) in zip(p0,diff)]
    b2 = list(map(sum,zip(*a2)))
    
    try:
        if foe == None:
            foe = np.linalg.solve([[b2[0],b2[1]],[b2[1],b2[2]]],[b2[3],b2[4]])
        else:
            foe = tuple(map(sum,zip(np.linalg.solve([[b2[0],b2[1]],[b2[1],b2[2]]],[b2[3],b2[4]]),foe)))
            #print(foe)
    except np.linalg.linalg.LinAlgError:
        foe = foe
    foe = (int(foe[0]/2+0.5), int(foe[1]/2 + 0.5))
    frame = cv2.circle(frame,foe,10,(0,255,0),-1)
    #img = cv2.add(frame,mask)

    
    cv2.imshow('frame',frame)    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    framecount += 1
    if framecount % 10 == 0:
        print("yay")
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    else:
        p0 = p1[st==1].reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()
