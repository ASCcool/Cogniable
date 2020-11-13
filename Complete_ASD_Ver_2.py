#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install keras-video-generators')
import os, uuid
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__
import pickle
connection_string = "DefaultEndpointsProtocol=https;AccountName=cogniableold;AccountKey=obxB3FMDXO3xz/pdO96V91Mki7xI1CKop9bhQkCdr5kdTqV8bmGXh15uBafQimQzKr2CjALO4FTUxB8+E2KWgg==;EndpointSuffix=core.windows.net"
container_name = "awsdata"
bucket = ContainerClient.from_connection_string(conn_str=connection_string, container_name=container_name)
#blob_list = bucket.list_blobs(name_starts_with="ai_team/rgb_data/Pytorch/Model v1")
#train_blob_list = bucket.walk_blobs(name_starts_with="ai_team/rgb_data/Data_v2/Train_data",include=None,delimiter='/')
video_list = bucket.list_blobs(name_starts_with="ASD")


# In[2]:


import os
import sys
import cv2
import time
import pickle
import numpy as np
import tensorflow as tf
from keras_video import VideoFrameGenerator
from datetime import datetime
from tensorflow.python.platform import app, flags
print('cv2 version: {}'.format(cv2.__version__))
#help(cv2.optflow)
#from cv2.optflow import DualTVL1OpticalFlow_create as DualTVL1
DualTVL1 = cv2.optflow.DualTVL1OpticalFlow_create()


# In[3]:


def formatFrames_rgb(frames,diagnosis):
    max_frames = 20
    group_frame = []
    batch_frame = []
    label = []
    frame_size = len(frames)
    print("No of frames:",frame_size)
    for i in range(1,frame_size):
        if i % max_frames != 0:
            group_frame.append(frames[i])
        elif (len(group_frame) >= 19):
            group_frame.append(frames[i])
            print("No of frames in each group:",len(group_frame))
            group_frame = np.reshape(group_frame,(1,20,224,224,3))
            print(group_frame.shape)
            batch_frame.append(group_frame)
            label.append(diagnosis)
            print("set complete:",len(group_frame))
            group_frame = []
    print("Total no of batches for the video file:",len(batch_frame))
    print("Corresponding labels:",len(label))
    return batch_frame, label


# In[4]:


def formatFrames_flow(frames,diagnosis):
    max_frames = 20
    group_frame = []
    batch_frame = []
    label = []
    frame_size = len(frames)
    print("No of frames:",frame_size)
    for i in range(1,frame_size):
        if i % max_frames != 0:
            group_frame.append(frames[i])
        elif (len(group_frame) >= 19):
            group_frame.append(frames[i])
            group_frame = np.reshape(group_frame,(1,20,224,224,2))
            print(group_frame.shape)
            batch_frame.append(group_frame)
            label.append(diagnosis)
            print("No of frames in each group:",len(group_frame))
            group_frame = []
    print("Total no of batches for the video file:",len(batch_frame))
    print("Corresponding labels:",len(label))
    return batch_frame, label


# In[5]:


all_videos = []
for blob in video_list:
  if blob.name.split(".")[-1].lower() == "mp4":
    all_videos.append(blob.name)
print(len(all_videos))
print(all_videos[0])


# In[6]:


class Person : 
    def __init__(self):
        self.path=None
        self.name=None
        self.gender=None
        self.diagnosis=None        
        self.play = []
        self.maladaptiveBehaviour=[]
        self.stereotypies = {
            "Motor":[],
            "Vocal":[],
            "Visual":[]
        }
        self.socialBehaviour = {
            "Facial Expression":[],
            "Eye Contact":[],
            "Social Interaction":[],
            "Social Greetings": []
        }
        self.jointAttention = {
            "Follows Gaze":[],
            "Follows a Point":[],
            "Shows objects spontaneously":[],
            "Expressive Communication ":[], 
            "Shares happiness":[]           
        }
        
        self.expressiveCommunication ={
            "Shake head for No":[],
            "Nods for Yes":[],
            "Asks for help":[],
            "Points to ask or choose":[],     
        }
        self.imitation ={
           "Actions on Objects":[],
            "Orofacial Imitation":[],
            "Motor Actions":[]
        }
        self.receptiveCommunication = {
            "Response to Name":[],
            "Response to Smile":[]
        }
        self.sensoryIssues={
            "Auditory":[],
            "Visual":[],
            "Texture":[]
        }
   
    def __str__(self):
        print("Name : ",self.name)
        print("Diagnosis: ",self.diagnosis)
        print("Gender: ", self.gender)
        print("Play : ")
        for p in self.play:
            print(p)
        
        print("\nStereotypies")
        print("*"*30)
        for i,j in self.stereotypies.items():
            print(f"{i}:{j}")
        
        print("\nSocial Behaviour")
        print("*"*30)
        for i,j in self.socialBehaviour.items():
            print(f"\n{i} :")
            for p in j:
                print(p)  
            
        return("")


# In[7]:


def dump_pickle(file_name, path, loss):
  with open(file_name, "wb") as f:
    pickle.dump(loss,f)
  blob_client = bucket.get_blob_client(path)
  with open(file_name,"rb") as data:
    blob_client.upload_blob(data)
  return


# In[8]:


def compute_rgb(video_file, time_stamp, diagnosis):

  """
    video file = the file to be processed
    time_stamp = tuple of form (start_time, end_time) in seconds
  """
  start = float(time_stamp[0])*1000
  end = float(time_stamp[1])*1000
  _IMAGE_SIZE= 224
  cap = cv2.VideoCapture(video_file)
  cap.set(cv2.CAP_PROP_POS_MSEC,start) #set the initial frame to start_time(in milli seconds)
  #fps = cap.get(cv2.CAP_PROP_FPS)
  #vid_len = fps*(float(time_stamp[1])-float(time_stamp[0]))    #find the total number of frames
  #vid_len = int(round(vid_len))
  rgb = []
  labels = []
  print("RGB",end="...")
  #cap.get(cv2.CAP_PROP_POS_MSEC)<=end
  while(cap.get(cv2.CAP_PROP_POS_MSEC)<=end):
      #print(cap.get(cv2.CAP_PROP_POS_MSEC))
      ret, frame2 = cap.read()
      if(ret==False):
              continue
      curr = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
      curr = cv2.resize(curr, (_IMAGE_SIZE, _IMAGE_SIZE))
      curr = np.array(curr,dtype = np.float64)
      curr = 2*(curr - np.min(curr))/np.ptp(curr)-1
      rgb.append(curr)
  train,label = formatFrames_rgb(rgb,diagnosis)
  cap.release()
  cv2.destroyAllWindows()
  #rgb = np.array(rgb, dtype = np.float64)
  return train,label


# In[9]:


def compute_flow(video_file, time_stamp,TVL1,diagnosis):
    start = float(time_stamp[0])*1000
    end = float(time_stamp[1])*1000
    _IMAGE_SIZE= 224
    cap = cv2.VideoCapture(video_file)
    cap.set(cv2.CAP_PROP_POS_MSEC,start)
    ret, frame1 = cap.read()
    flow = []
    labels = []
    if(ret==False):
        return flow
    prev = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    prev = cv2.resize(prev, (_IMAGE_SIZE, _IMAGE_SIZE))
    print("FLOW",end="...")
    while(cap.get(cv2.CAP_PROP_POS_MSEC)<=end):
        #print(cap.get(cv2.CAP_PROP_POS_MSEC))
        ret, frame2 = cap.read()
        if(ret==False):
                continue
        curr = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        curr = cv2.resize(curr, (_IMAGE_SIZE, _IMAGE_SIZE))
        curr_flow = TVL1.calc(prev, curr, None)
        assert(curr_flow.dtype == np.float32)
        #print(prev.shape)
    # truncate [-20, 20]
        curr_flow[curr_flow >= 20] = 20
        curr_flow[curr_flow <= -20] = -20
    # scale to [-1, 1]
        #max_val = lambda x: max(max(x.flatten()), abs(min(x.flatten())))
        #curr_flow = curr_flow / max_val(curr_flow)
        curr_flow = 2*(curr_flow - np.min(curr_flow))/np.ptp(curr_flow)-1
        flow.append(curr_flow)
        prev = curr
    train,label = formatFrames_flow(flow,diagnosis)
    cap.release()
    cv2.destroyAllWindows()
    #flow = np.array(flow)
    return train,label


# In[10]:


import io
from io import BytesIO
path = f"Final_Binary_Classification_Dataset/child_data.pkl"
file_name = io.BytesIO(bucket.download_blob(path).readall())
anvil_objects = pickle.load(file_name)
name_indexed_video_path={}
for o in anvil_objects:
  for key in all_videos:
    if(o.path.lower() == key.split("/")[-1].lower()):
      name_indexed_video_path[o.path]=key
print(len(name_indexed_video_path))
print(name_indexed_video_path)


# In[11]:


attributes = [("socialBehaviour","Eye Contact"),
("jointAttention","Follows a Point"),
("expressiveCommunication","Nods for Yes"),
("imitation","Orofacial Imitation"),
("receptiveCommunication","Response to Name"),
("sensoryIssues","Visual"),
("stereotypies","Vocal")]


# In[12]:


def process_video(video_name, anvil_obj, global_count_rgb, global_count_flow):
  TVL1 = DualTVL1
  print("============processing"+ video_name +"================")
  #global_count = 0
  for i in range(len(attributes)):
    attrib = getattr(anvil_obj, attributes[i][0])
    print(attributes[i][0]+" "+attributes[i][1])
    rgb=[]
    flow=[]
    rgb_labels=[]
    flow_labels=[]
    j=0
    ################################################################
    for elem in attrib[attributes[i][1]]:
      print(elem)
      print("instance "+str(j+1),end="===")
      t0=time.clock()
      #frames,labels =compute_rgb(video_name, elem['time stamps'],anvil_obj.diagnosis) #changed -> each frame should have a label
      #with this line a each set of timestamps will have a label not for every frames
      if(anvil_obj.diagnosis!='None'):
        rgb_data,labels_rgb =compute_rgb(video_name, elem['time stamps'],anvil_obj.diagnosis)
        t1=time.clock()
        print("time taken="+str(round(t1-t0,3)), end=" seconds===")
        rgb = rgb + rgb_data
        rgb_labels = rgb_labels + labels_rgb
        t0=time.clock()
        flow_data,labels_flow = compute_flow(video_name, elem['time stamps'], TVL1,anvil_obj.diagnosis)#changed -> each frame should have a label
        #with this line a each set of timestamps will have a label not for every frames
        flow = flow + flow_data
        flow_labels = flow_labels + labels_flow #this should be done for all the frames, so it needs to be inside the compute_rgb and compute_flow
        t1=time.clock()
        print("time taken="+str(round(t1-t0,3))+" seconds")
        j=j+1

    if(len(rgb)>0):
      global_count_rgb += 1;
      rgb_key = f"Final_Binary_Classification_Dataset/rgb_data/v5.0/ASD/train_{global_count_rgb}.pkl" #for ASD -> change it to Neurotypical for NT
      dump_pickle("temp_NT.pkl", rgb_key, [rgb,rgb_labels]) # [X,y] #changed -> this create one temp instance and uploads all the data to the destination
    
    if(len(flow)>0):
      global_count_flow += 1;
      flow_key = f"Final_Binary_Classification_Dataset/flow_data/v5.0/ASD/train_{global_count_flow}.pkl" #for ASD -> change it to Neurotypical for NT
      dump_pickle("temp_flow_NT.pkl", flow_key, [flow,flow_labels]) # [X,y] #changed -> this create one temp instance and uploads all the data to the destination
    
    del flow
    del rgb
    print("Attribute DONE!")
  return global_count_rgb, global_count_flow


# In[13]:


count =0
global_count_rgb = 0
global_count_flow = 0
for i in range(len(anvil_objects)):
  t0=time.clock()
  if(anvil_objects[i].path not in name_indexed_video_path.keys()):
    continue
  vid_name = anvil_objects[i].path
  print(vid_name)
  vid_key = name_indexed_video_path[vid_name]
  print(vid_key)
  vid_file = bucket.get_blob_client(blob).download_blob().readall()
  with open(vid_name, "wb") as file:
      file.write(vid_file)
  global_count_rgb, global_count_flow = process_video(vid_name, anvil_objects[i], global_count_rgb, global_count_flow)
  print("Global count rgb = "+str(global_count_rgb))
  print("Global count flow = "+str(global_count_flow))
  os.remove(vid_name)
  t1=time.clock()
  print(i)
  count+=1
  print ("************* processed ="+str(count)+" ********************")
  print("Total time taken="+str(round((t1-t0)/60,3))+" minutes")

