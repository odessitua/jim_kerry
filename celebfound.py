# -*- coding: utf-8 -*-
from deepface import DeepFace
import gdown
import cv2
import os
import numpy as np
import time
import pandas as pd
from PIL import Image
from IPython.display import clear_output

class CelebFound(object):
  def __init__(self,fileId='1ETKBfNoXO8Kafr8Kn1tbzozb94ygRI41',fileName=None,pathToIds='kerry.ids.npy',dbPath = "kerry"):
    """Init class CelebFound

    Parameters
    ----------
    fileId: str
        id video in google Drive https://drive.google.com/uc?id={}'.format(fileId)
    fileName: str, optional
        path to video file, if not in google Drive. Default is None.
    pathToIds: str
        path to npy file, where ids of frames will be saving. Default is kerry.ids.npy.
    dbPath: str
        path to folder where are stored the photos of celebrity or file representation - /representations_vgg_face.pkl. Default is 'kerry'.
    """
    self.imgsPath = 'videoImgs/' # tmp folder for frames
    os.makedirs(self.imgsPath, exist_ok=True)

    self.pathToIds = pathToIds
    self.dbPath = dbPath
    

    if (fileName!=None):
      self.fileName = fileName
    else:
      self.fileName = gdown.download('https://drive.google.com/uc?id={}'.format(fileId), None, quiet=True)

    try:
      self.frameIds = np.load(self.pathToIds).tolist()
      print("Load ids count:",len(self.frameIds),'Last:',self.frameIds[-1])
    except:
      self.frameIds = []
    
    self.fastMode = False

  def searchFrames(self,minSimilar=0, secToEnd=None, fastMode=False):
    """Search frames where celeb is

    Parameters
    ----------
    minSimilar: int, optional
        the degree of recognition. How many from the base the celebrity's photo 
        must match in order for the frame to be preserved. Default is 0.
    secToEnd: int, optional
        second to break loading. Use if you dont want checking all video. Default is 0.
    fastMode: bool
        more speed worse result. Default is False.
    """
    self.fastMode = fastMode
    cap = cv2.VideoCapture(self.fileName)
    if (fastMode):
      step = 10
    else:
      step = 1

    # if ids have already been saved, then use the last frame as the beginning
    if self.frameIds:
      lastId = self.frameIds[-1] + step
      cap.set(cv2.CAP_PROP_POS_FRAMES,lastId)
    else:
      lastId=0

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    if (secToEnd!=None):
      length = secToEnd*fps

    for i in range(lastId,length,step):
      if (self.fastMode):
        cap.set(cv2.CAP_PROP_POS_FRAMES,i)
      #read frame
      ret, frame = cap.read()

      #save frame or break loading if end
      if (ret):
        imgName = self.imgsPath + str(i) + ".jpg"
        cv2.imwrite(imgName, frame)
      else:
        break 

      #search celeb in frame
      try:
        df = DeepFace.find(imgName, db_path = self.dbPath)
        if (df.shape[0]>minSimilar):
          if (fastMode):
            start = min(i,abs(i-5))
            for j in range(start,i+5):
              self.frameIds.append(j)
          else:
            self.frameIds.append(i)
      except:
        pass

      clear_output()  
      #every 100 frames save data
      if (i%100==0):
        np.save(self.pathToIds,np.array(self.frameIds))

      print("Frame:",i,'/',length,'. Found celeb:',len(self.frameIds))
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    np.save(self.pathToIds,np.array(self.frameIds))
    cap.release()
    cv2.destroyAllWindows()

  def clearFrameIds(self):
    """clear frameIds"""
    self.frameIds = []

  def fillMissFrames(self):
    """Fill missing frames"""
    
    if (self.fastMode):
      step = 21
    else:
      step = 10
    newFrameIds = []
    for id,val in enumerate(self.frameIds[:-1]):
      newFrameIds.append(val)
      div = self.frameIds[id+1] - val
      if (div > 1) & (div <= step):
        for i in range(val+1,self.frameIds[id+1]):
          newFrameIds.append(i)
    newFrameIds.append(self.frameIds[id+1])
    print("Old frames count:",len(self.frameIds))
    self.frameIds = newFrameIds
    print("New frames count:",len(self.frameIds))

  def countCelebTime(self):
    """Сount the time how many celebrities were on the screen

    Returns
    ----------
    In seconds
    :(celeb time, length of video)
    """
    cap = cv2.VideoCapture(self.fileName)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    celebTime = round(len(self.frameIds)/fps)
    lengthTime = round(length/fps)
    return celebTime, lengthTime

  def saveCelebVideo(self,outputPath='output.mp4'):
    """Save video

    Parameters
    ----------
    outputPath: str, optional
        path to output video file. Default is output.mp4.
    """
    cap = cv2.VideoCapture(self.fileName)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outputPath, fourcc, 20.0, (width, height))
    for i in range(self.frameIds[-1]+1):
      ret, frame = cap.read()
      if i in self.frameIds:
        out.write(frame)
    out.release()
    cap.release()