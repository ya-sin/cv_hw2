import cv2
import numpy as np
import tkinter as tk
import PIL.Image, PIL.ImageTk
from matplotlib import pyplot as plt

fn1 = './data/Bird1.jpg'
fn2 = './data/Bird2.jpg'
fn3 = './data/bgSub.mp4'
fn4 = './data/featureTracking.mp4'
fn5 = './data/feature.flv'

def fun1_1():
  bird1 = cv2.imread(fn1)
  gray1= cv2.cvtColor(bird1,cv2.COLOR_BGR2GRAY)
  # construct a SIFT object
  sift1 = cv2.xfeatures2d.SIFT_create()
  # finds the keypoint
  kp1, des1 = sift1.detectAndCompute(gray1,None)
  # find the feature point at P(179.9, 114.0)
  i = 0
  while 1:
    i = i+1
    if(round(kp1[i-1].pt[0],1) == 179.9):
      print(kp1[i-1].angle)
      break
  # draw the keypoint P(179.9, 114.0)
  img1=cv2.drawKeypoints(gray1,kp1[i-1:i],bird1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
  # plot the result
  des_list =  des1[i-1]

  plt.subplot(1,2,1),plt.imshow(img1)
  plt.title('bird1'), plt.xticks([]), plt.yticks([])
  plt.subplot(1,2,2),plt.bar(range(len(des_list)), height=des_list, width=0.4, alpha=0.8, color='blue')
  plt.ylim(0, 180)
  plt.title('featureVectorHistogram')
  plt.show()

def fun1_2():
  bird1 = cv2.imread(fn1)
  bird2 = cv2.imread(fn2)
  gray1= cv2.cvtColor(bird1,cv2.COLOR_BGR2GRAY)
  gray2= cv2.cvtColor(bird2,cv2.COLOR_BGR2GRAY)
  # construct a SIFT object
  sift1 = cv2.xfeatures2d.SIFT_create()
  sift2 = cv2.xfeatures2d.SIFT_create()
  # finds the keypoint
  kp1, des1 = sift1.detectAndCompute(gray1,None)
  kp2, des2 = sift2.detectAndCompute(gray2,None)
  # print(kp1[0].pt)
  img1=cv2.drawKeypoints(gray1,kp1[213:219],bird1)
  img2=cv2.drawKeypoints(gray2,kp2[214:220],bird2)
  # save the image
  cv2.imwrite('FeatureBird1.jpg',img1)
  cv2.imwrite('FeatureBird2.jpg',img2)
  # show the result
  cv2.imshow('result1',np.hstack((img1,img2)))
  cv2.waitKey(0)
  cv2.destroyAllWindows()
def fun1_3():
  bird1 = cv2.imread(fn1)
  bird2 = cv2.imread(fn2)
  gray1= cv2.cvtColor(bird1,cv2.COLOR_BGR2GRAY)
  gray2= cv2.cvtColor(bird2,cv2.COLOR_BGR2GRAY)
  # construct a SIFT object
  sift1 = cv2.xfeatures2d.SIFT_create()
  sift2 = cv2.xfeatures2d.SIFT_create()
  # finds the keypoint
  kp1, des1 = sift1.detectAndCompute(gray1,None)
  kp2, des2 = sift2.detectAndCompute(gray2,None)
  test1 = des1[213:219]
  test2 = des2[214:220]
  # BFMatcher with default params
  bf = cv2.BFMatcher()
  matches = bf.knnMatch( test1, test2, k=2 )
  # Apply ratio test
  good = []
  i = 0
  for m,n in matches:
      i = i+1
      if m.distance < 0.75*n.distance:
          good.append([m])
  # cv.drawMatchesKnn expects list of lists as matches.
  img3 = cv2.drawMatchesKnn(gray1,kp1[213:219],gray2,kp2[214:220],good,None,flags=2)
  plt.axis("off")
  plt.imshow(img3)
  plt.show()

def fun2_1():
  cap = cv2.VideoCapture(fn3)
  fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(50, 3, 0.8,0)

  while(1):
      ret, frame = cap.read()
      fgmask = fgbg.apply(frame)
      cv2.imshow('test1',frame)
      cv2.imshow('test2',fgmask)
      k = cv2.waitKey(30) & 0xff
      if k == 27:
          break
  cap.release()
  cv2.destroyAllWindows()

def fun3_1():
  cap = cv2.VideoCapture(fn4,0)

  ret, frame = cap.read()
  if not ret:
      pass

  frame = cv2.convertScaleAbs(frame)

  # setting parameter
  param = cv2.SimpleBlobDetector_Params()

  param.minDistBetweenBlobs = 18
  param.filterByConvexity = True
  param.filterByCircularity = True
  param.minCircularity = 0.84
  param.filterByArea = True
  param.minArea = 30
  param.maxArea = 80

  # detect keypoint
  detect = cv2.SimpleBlobDetector_create(param)
  kp = detect.detect(frame)

  img = frame.copy()

  # show squares on images
  if(ret):
      for i in range(0,len(kp)):
          x,y = np.int(kp[i].pt[0]),np.int(kp[i].pt[1])
          size = np.int(kp[i].size)
          if size > 1:
              size = np.int(size/2)
          img = cv2.rectangle(img, (x-size,y-size), (x+size,y+size), (0,0,255), thickness=-1)
      cv2.imshow('Tracking whole video', img)
  else:
      pass
  cap.release()


def fun3_2():
  # Seven points
  cap = cv2.VideoCapture(fn5)

  # Parameters for lucas kanade optical flow
  lk_params = dict( winSize  = (21,21),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

  # Obtain the first frame
  ret, previous_frame = cap.read()
  if not ret:
      pass

  previous_frame = cv2.convertScaleAbs(previous_frame)

  # setting parameter
  params = cv2.SimpleBlobDetector_Params()
  params.minDistBetweenBlobs = 18
  params.filterByConvexity = True

  params.filterByCircularity = True
  params.minCircularity = 0.84

  params.filterByArea = True
  params.minArea = 30
  params.maxArea = 80

  # detect keypoint
  detect = cv2.SimpleBlobDetector_create(params)
  kp = detect.detect(previous_frame)
  p0 = []
  a0 = np.array([[np.float32(kp[0].pt[0]),np.float32(kp[0].pt[1])]])
  a1 = np.array([[np.float32(kp[1].pt[0]),np.float32(kp[1].pt[1])]])
  a2 = np.array([[np.float32(kp[2].pt[0]),np.float32(kp[2].pt[1])]])
  a3 = np.array([[np.float32(kp[3].pt[0]),np.float32(kp[3].pt[1])]])
  a4 = np.array([[np.float32(kp[4].pt[0]),np.float32(kp[4].pt[1])]])
  a5 = np.array([[np.float32(kp[5].pt[0]),np.float32(kp[5].pt[1])]])
  a6 = np.array([[np.float32(kp[6].pt[0]),np.float32(kp[6].pt[1])]])
  p0 = np.array([a0, a1, a2, a3, a4, a5, a6])

  previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)

  # Create a mask image for drawing purposes
  mask = np.zeros_like(previous_frame)
  while(1):
      ret,frame = cap.read()
      gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

      # calculate optical flow
      p1, st, err = cv2.calcOpticalFlowPyrLK(previous_gray, gray_frame, p0, None, **lk_params)

      # Select good points
      good_new = p1[st==1]
      good_old = p0[st==1]

      # draw the tracks
      for i,(new,old) in enumerate(zip(good_new,good_old)):
          a,b = new.ravel()
          c,d = old.ravel()
          mask = cv2.line(mask, (a,b),(c,d), (0, 0, 255), 2)
          frame = cv2.circle(frame,(a,b),5,(0,0,255),-1)
      img = cv2.add(frame,mask)

      cv2.imshow('frame',img)
      k = cv2.waitKey(30) & 0xff
      if k == 27:
          break

      # Now update the previous frame and previous points
      previous_gray = gray_frame.copy()
      p0 = good_new.reshape(-1,1,2)

  cv2.destroyAllWindows()
  cap.release()

main_window = tk.Tk()
main_window.geometry('1080x720')
main_window.title('HW2')

# all frames
left_frame = tk.Frame(main_window)
left_frame.place(relwidth=0.3, relheight=1)
middle_frame = tk.Frame(main_window)
middle_frame.place(relx=0.3, relwidth=0.3, relheight=1)
right_frame = tk.Frame(main_window)
right_frame.place(relx=0.6, relwidth=0.4, relheight=1)

# all labels
l1 = tk.Label(left_frame, text='1. SIFT', font=("Helvetica", 24))
l1.grid(row=0, sticky='w', padx=5, pady=30)
l2 = tk.Label(left_frame, text='2. Background Subtraction', font=("Helvetica", 24))
l2.grid(row=5, sticky='w', padx=5, pady=30)
l3 = tk.Label(middle_frame, text='3. Feature Tracking', font=("Helvetica", 24))
l3.grid(row=0, sticky='w', padx=5, pady=30)

# all buttons
b11 = tk.Button(left_frame, text='1.1 Feature Patch', font=("Helvetica", 18), command=fun1_1)
b11.grid(row=1, sticky='nsew', padx=20, pady=20)
b12 = tk.Button(left_frame, text='1.2 Feature Points', font=("Helvetica", 18), command=fun1_2)
b12.grid(row=2, sticky='nsew', padx=20, pady=20)
b13 = tk.Button(left_frame, text='1.3 Match Feature Points', font=("Helvetica", 18), command=fun1_3)
b13.grid(row=3, sticky='nsew', padx=20, pady=20)
b21 = tk.Button(left_frame, text='2.1 Background Subtractio', font=("Helvetica", 18), command=fun2_1)
b21.grid(row=6, sticky='nsew', padx=20, pady=20)
b31 = tk.Button(middle_frame, text='3.1 Preprocessing', font=("Helvetica", 18), command=fun3_1)
b31.grid(row=1, sticky='nsew', padx=20, pady=20)
b32 = tk.Button(middle_frame, text='3.2 Video Tracking', font=("Helvetica", 18), command=fun3_2)
b32.grid(row=2, sticky='nsew', padx=20, pady=20)

main_window.mainloop()
