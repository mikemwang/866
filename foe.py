import numpy as np
import cv2
import motion
import matplotlib
import matplotlib.pyplot as plt
import yaml
import sys

# select which one to activate

#cap = cv2.VideoCapture('mp4_movies/train.mp4')
cap = cv2.VideoCapture(sys.argv[1])


# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 500,
                       qualityLevel = 0.000001,
                       minDistance = 1,
                       blockSize = 7,
                       gradientSize = 3,
                       mask = None)

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, **feature_params)
foe = None
framecount = 0

### TTC VARIABLES BEGIN #############
ttc = motion.TTC()
data = []
### TTC VARIABLES END #############

while(1):
    ret,frame = cap.read()
    if not ret:
        break
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.blur(frame_gray, (5,5))

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
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

    ###   TTC CODE BEGIN ################
    if not ttc.init_done:
        ttc.init_on_first_frame(old_gray)
    if int(sys.argv[2]) == 1:
        ttc.update(old_gray, frame_gray, (foe[0]-frame.shape[1]//2,
                  foe[1]-frame.shape[0]//2))
    else:
        ttc.update(old_gray, frame_gray)
    data.append(float(ttc.t))
    ###   TTC CODE END ##################


    frame = cv2.circle(frame_gray,foe,10,(255,255,0),-1)
    #mask = abs(ttc.gradient.Et) > 0.1
    #mask = np.array(mask, dtype=np.uint8)
    #e = mask[:,-1]
    #e.shape = [mask.shape[0], 1]
    #mask = np.hstack((mask, e))

    #e = mask[-1,:]
    #e.shape = [1,mask.shape[1]]
    #mask = np.vstack((mask, e))

    #feature_params['mask'] = mask
    #img = cv2.add(frame,mask)


    cv2.imshow('frame',frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    framecount += 1
    if framecount % 10 == 0:
        p0 = cv2.goodFeaturesToTrack(old_gray, **feature_params)
    else:
        p0 = p1[st==1].reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()


### TTC VISUALIZATION ##############
#stream = file('experimental_data/data.yaml', 'w')
with open('experimental_data/data.yaml', 'r') as f:
    data_file = yaml.load(f)
experiment_name = sys.argv[1]
sub_ex = "NO_FOE" if int(sys.argv[2]) == 0 else "WITH_FOE"

if experiment_name not in data_file:
    data_file[experiment_name] = {"WITH_FOE": dict(), "NO_FOE": dict()}
xs = [i for i in range(len(data))]
data_file[experiment_name][sub_ex]['frames'] = xs
data_file[experiment_name][sub_ex]['data'] = data
data_file[experiment_name][sub_ex]['groundtruth'] = xs[::-1]

with open('experimental_data/data.yaml', 'w') as f:
    yaml.dump(data_file, f)

fig, ax = plt.subplots()
ax.plot(xs, data)
ax.plot(xs, xs[::-1])
plt.show()
### TTC VISUALIZATION ##############
