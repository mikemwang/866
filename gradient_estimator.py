import numpy as np
import cv2
import time
import sys

cap = cv2.VideoCapture('ttc_real.mov')
last_gray = None

class GradientEstimator:

    def __init__(self):
        self.dx = 1
        self.dy = 1
        self.dt = 1
        self.Ex = None
        self.Ey = None
        self.Et = None

    def update(self, old_frame, new_frame):
        n, m = old_frame.shape
        old_frame = old_frame.astype(float)
        new_frame = new_frame.astype(float)

        # at k:   A B
        #         C D
        #
        # at k+1:
        #         W X
        #         Y Z
        #
        # want to evaluate: B + D + X + Z - (A + C +W +Y)
        #    equivalent to: B - A + D - C + (X - W + Z - Y)
        # so we can evaluate the quantity for each matrix first, then add
        #
        # Consider matrix k. We can evaluate its quantity by:
        # [delete first column] - [delete last column]
        # = B - A
        #   D - C
        # a 2 x 1 matrix, let's call this J. Then:
        #
        # [delete first row of J] + [delete last row of J]
        # = B - A + D - C
        J1 = np.delete(old_frame, 0, 1) - np.delete(old_frame, m-1, 1)
        k0 = np.delete(J1, n-1, 0) + np.delete(J1, 0, 0)

        # do the same for the matrix at k + 1
        J2 = np.delete(new_frame, 0, 1) - np.delete(new_frame, m-1, 1)
        k1 = np.delete(J2, n-1, 0) + np.delete(J2, 0, 0)

        # by the lecture notes
        self.Ex = 1.0/(4.0*self.dx)*(k0 + k1 )

        # for Ey, do the same thing except delete columns instead of rows
        J1 = np.delete(old_frame, 0, 0) - np.delete(old_frame, m-1, 0)
        k0 = np.delete(J1, n-1, 1) + np.delete(J1, 0, 1)
        J2 = np.delete(new_frame, 0, 0) - np.delete(new_frame, m-1, 0)
        k1 = np.delete(J2, n-1, 1) + np.delete(J2, 0, 1)
        self.Ey = 1.0/(4.0*self.dy)*(k0 + k1)

        # for Et, sum then subtract
        J1 = np.delete(old_frame, 0, 0) + np.delete(old_frame, m-1, 0)
        k0 = np.delete(J1, n-1, 1) + np.delete(J1, 0, 1)
        J2 = np.delete(new_frame, 0, 0) + np.delete(new_frame, m-1, 0)
        k1 = np.delete(J2, n-1, 1) + np.delete(J2, 0, 1)
        self.Et = 1.0/(4.0*self.dt)*(k1 - k0)


class TTC:
    def __init__(self):
        self.img_w = None
        self.img_h = None
        self.xs = None
        self.ys = None
        self.gradient = GradientEstimator()
        self.xEx = None
        self.yEy = None
        self.init_done = False
        self.t = 0

    def init_on_first_frame(self, frame):
        self.img_h, self.img_w = frame.shape
        # for calculating xEx
        self.xs = np.repeat(np.array([[i - (self.img_w-1)//2 for i in range(self.img_w-1)]]), self.img_w-1, axis=0)
        # for yEy
        self.ys = np.repeat(np.array([[i - (self.img_h-1)//2 for i in
            range(self.img_h-1)]]).T, self.img_h-1, axis=1)

        self.init_done = True

    def update(self, old_frame, new_frame):
        self.gradient.update(old_frame, new_frame)
        self.xEx = np.multiply(self.xs, self.gradient.Ex)
        self.yEy = np.multiply(self.ys, self.gradient.Ey)
        G_sq = - np.sum(self.xEx**2) + \
                 2*np.sum(np.multiply(self.xEx,self.yEy)) + \
                 np.sum(self.yEy**2)
        GEt = np.sum(np.multiply(self.xEx, self.gradient.Et)) + \
                np.sum(np.multiply(self.yEy, self.gradient.Et))
        self.t = - G_sq / GEt
        print(self.t)



ttc = TTC()
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if last_gray is None:
        ttc.init_on_first_frame(gray)
    else:
        ttc.update(last_gray, gray)


    last_gray = gray

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

#if __name__ == "__main__":
#    test = GradientEstimator()
#
#    test.update(np.array([[1, 1, 0],[1, 1, 0],[0, 0, 0]]),
#                np.array([[0, 0, 0],[0, 1, 1],[0, 1, 1]]))
#
