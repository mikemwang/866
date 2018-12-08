import numpy as np
import cv2
import time
import sys

cap = cv2.VideoCapture('ttc_small.mov')
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


gradient = GradientEstimator()

while(True):
    # Capture frame-by-frame
    t = time.time()
    while (time.time() - t < 1):
        pass
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if last_gray is not None:
        gradient.update(last_gray, gray)
        xs = np.array([[i - len(gradient.Ex[0])//2 for i in range(len(gradient.Ex[0]))]])
        ys = np.array([[i - len(gradient.Ex)//2 for i in range(len(gradient.Ex))]])
        print(xs, ys)

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
