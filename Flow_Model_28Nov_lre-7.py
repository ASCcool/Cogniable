#!/usr/bin/env python
# coding: utf-8

# In[1]:

#import logging

#logging.basicConfig(filename='output_final.log', encoding='utf-8', level=logging.INFO)

'''logging.basicConfig(handlers=[logging.FileHandler(filename="output_final.log", 
                                                 encoding='utf-8', mode='a+')],
                    format="%(asctime)s %(name)s:%(levelname)s:%(message)s", 
                    datefmt="%F %A %T", 
                    level=logging.INFO)'''

'''root_logger= logging.getLogger()
root_logger.setLevel(logging.INFO) # or whatever
handler = logging.FileHandler('output_final.log', 'w', 'utf-8') # or whatever
handler.setFormatter(logging.Formatter('%(name)s %(message)s')) # or whatever
root_logger.addHandler(handler)'''

#logging.basicConfig(filename='LOGGER.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

import os, uuid
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__
import pickle
from sklearn.utils import shuffle
connection_string = "DefaultEndpointsProtocol=https;AccountName=cogniableold;AccountKey=obxB3FMDXO3xz/pdO96V91Mki7xI1CKop9bhQkCdr5kdTqV8bmGXh15uBafQimQzKr2CjALO4FTUxB8+E2KWgg==;EndpointSuffix=core.windows.net"
container_name = "awsdata"
bucket = ContainerClient.from_connection_string(conn_str=connection_string, container_name=container_name)

import os
import io
import random

dataset = []
dataset_list = bucket.list_blobs(name_starts_with="Final_Combined_Model_Dataset/Modified_data_flow_V2")

for i in dataset_list:
    if i.name.split(".")[-1].lower() == "pkl":
        dataset.append(i.name)
random.shuffle(dataset)
samples = len(dataset)
train_set = int(0.6 * len(dataset)) # 60% for training
val_set = len(dataset) - train_set # 40% for validation
train_data = dataset[:train_set]
val_data = dataset[train_set:]
print("Training data: ",train_data)
print("Validation data: ",val_data)
print("No of training data: ",len(train_data))
print("No of validation data: ",len(val_data))

def get_train_data(a):          
    data = train_data[a]
    print(data)
    temp = io.BytesIO(bucket.download_blob(data).readall())
    temp1 = []
    temp1 = pickle.load(temp) 
    return temp1

def get_val_data(a):          
    data = val_data[a]
    print(data)
    temp = io.BytesIO(bucket.download_blob(data).readall())
    temp1 = []
    temp1 = pickle.load(temp) 
    return temp1


    
def get_labels(label):
    order_label=['Neurotypical','ASD']
    y = []
    for i in label:
        y.append(order_label.index(i))
    return y


# In[2]:


import torch.nn as nn
import torch

class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=False)  # verify bias false

        # verify defalt value in sonnet
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Mixed_3b(nn.Module):
    def __init__(self):
        super(Mixed_3b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(192, 64, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(192, 96, kernel_size=1, stride=1),
            BasicConv3d(96, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(192, 16, kernel_size=1, stride=1),
            BasicConv3d(16, 32, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(192, 32, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)

        return out


class Mixed_3c(nn.Module):
    def __init__(self):
        super(Mixed_3c, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv3d(256, 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(256, 128, kernel_size=1, stride=1),
            BasicConv3d(128, 192, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(256, 32, kernel_size=1, stride=1),
            BasicConv3d(32, 96, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(256, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4b(nn.Module):
    def __init__(self):
        super(Mixed_4b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(480, 192, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(480, 96, kernel_size=1, stride=1),
            BasicConv3d(96, 208, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(480, 16, kernel_size=1, stride=1),
            BasicConv3d(16, 48, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(480, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4c(nn.Module):
    def __init__(self):
        super(Mixed_4c, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 160, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 112, kernel_size=1, stride=1),
            BasicConv3d(112, 224, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 24, kernel_size=1, stride=1),
            BasicConv3d(24, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4d(nn.Module):
    def __init__(self):
        super(Mixed_4d, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 128, kernel_size=1, stride=1),
            BasicConv3d(128, 256, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 24, kernel_size=1, stride=1),
            BasicConv3d(24, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4e(nn.Module):
    def __init__(self):
        super(Mixed_4e, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 112, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 144, kernel_size=1, stride=1),
            BasicConv3d(144, 288, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 32, kernel_size=1, stride=1),
            BasicConv3d(32, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4f(nn.Module):
    def __init__(self):
        super(Mixed_4f, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(528, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(528, 160, kernel_size=1, stride=1),
            BasicConv3d(160, 320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(528, 32, kernel_size=1, stride=1),
            BasicConv3d(32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(528, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_5b(nn.Module):
    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(832, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(832, 160, kernel_size=1, stride=1),
            BasicConv3d(160, 320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(832, 32, kernel_size=1, stride=1),
            BasicConv3d(32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(832, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_5c(nn.Module):
    def __init__(self):
        super(Mixed_5c, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(832, 384, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(832, 192, kernel_size=1, stride=1),
            BasicConv3d(192, 384, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(832, 48, kernel_size=1, stride=1),
            BasicConv3d(48, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=1, padding=1),
            BasicConv3d(832, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out

class I3D(nn.Module):

    def __init__(self, num_classes=1, dropout_drop_prob = 0.5, input_channel = 2, spatial_squeeze=True):
        super(I3D, self).__init__()
        self.features = nn.Sequential(
            BasicConv3d(input_channel, 64, kernel_size=7, stride=2, padding=3), # (64, 32, 112, 112)
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),  # (64, 32, 56, 56)
            BasicConv3d(64, 64, kernel_size=1, stride=1), # (64, 32, 56, 56)
            BasicConv3d(64, 192, kernel_size=3, stride=1, padding=1),  # (192, 32, 56, 56)
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),  # (192, 32, 28, 28)
            Mixed_3b(), # (256, 32, 28, 28)
            Mixed_3c(), # (480, 32, 28, 28)
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)), # (480, 16, 14, 14)
            Mixed_4b(),# (512, 16, 14, 14)
            Mixed_4c(),# (512, 16, 14, 14)
            Mixed_4d(),# (512, 16, 14, 14)
            Mixed_4e(),# (528, 16, 14, 14)
            Mixed_4f(),# (832, 16, 14, 14)
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)), # (832, 8, 7, 7)
            Mixed_5b(), # (832, 8, 7, 7)
            Mixed_5c(), # (1024, 8, 7, 7)
            nn.AvgPool3d(kernel_size=(2, 7, 7), stride=1),# (1024, 8, 1, 1)
            nn.Dropout3d(dropout_drop_prob),
            nn.Conv3d(1024, num_classes, kernel_size=1, stride=1, bias=True),# (400, 8, 1, 1)
        )
        self.spatial_squeeze = spatial_squeeze
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        logits = self.features(x)

        if self.spatial_squeeze:
            logits = logits.squeeze(3)
            logits = logits.squeeze(3)

        averaged_logits = torch.mean(logits, 2)
        predictions = self.sigmoid(averaged_logits)

        return predictions, averaged_logits


# In[3]:


def get_model_flow(a = ''):
    print(a)
    blob_list = bucket.list_blobs(name_starts_with="Final_Binary_Classification_Dataset/Flow_Model_28Nov_lre-7/")
    path = ""
    for blob in blob_list:      
        if (a != '') and blob.name == f"Final_Binary_Classification_Dataset/Flow_Model_28Nov_lre-7/FlowModel_{a}.pkl":
          path = blob.name
          print(path)
          break
    if path == "":
        print("Model Not found")
        return
    #temp= "RGB Model"   
    #file = bucket.download_blob(path)
    #stream = file.readinto(temp)
    temp1 = I3D(num_classes= 2)
    temp1 = torch.nn.DataParallel(temp1)
    temp1.load_state_dict(torch.load(io.BytesIO(bucket.download_blob(path).readall()),map_location={'cuda:0':'cpu'}),strict=False)
    print("Model State Dict:",temp1)
    return temp1   


# In[4]:


#flow = get_model_flow(413)
#flow = I3D(num_classes= 1)
flow = I3D(num_classes= 2)


# In[5]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
flow.to(device)


# In[6]:


import time
t = []
asd = []


# In[7]:


import numpy as np
from torch.autograd import Variable
def transform(x):
    s = torch.from_numpy(np.array(x).reshape(1,20,224,224,2))
    s = Variable(s.permute(0, 4, 1, 2 ,3))
    return s


# In[8]:


def plot_loss(tem):   
    import matplotlib.pyplot as plt
    plt.figure(figsize=(30,5))
    plt.plot(tem)
    plt.savefig('Final_loss_plot.jpg')
    plt.show()


# In[9]:


def dump_pickle(file_name, path, loss):
  with open(file_name, "wb") as f:
    pickle.dump(loss,f)
  blob_client = bucket.get_blob_client(path)
  with open(file_name,"rb") as data:
    blob_client.upload_blob(data)
  return

def dump(file_name, path, model):
  torch.save(model.state_dict(), file_name)
  blob_client = bucket.get_blob_client(path)
  with open(file_name,"rb") as data:
    blob_client.upload_blob(data)
  return


# In[10]:


import random
def get_index(n):
  i = list(range(1,n))
  random.shuffle(i)
  return i


# In[11]:


import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

criterion = nn.CrossEntropyLoss()

#optimizer = optim.Adam(flow.parameters(), lr=0.000001)


# In[12]:


def calculate_train_accuracy():    
    global x
    global y
    global flow
    samples = 0
    count = 0
    incorrect = 0
    c_CF = []
    y_CF = []
    flow.eval()
    tq = get_index(len(train_data))
    #random.shuffle(train_data)
    print("Calculating train accuracy....") 
    with torch.no_grad():
      for j in tq:
          c = []
          x,y = get_train_data(j)
          y = get_labels(y)
          samples += len(y)
          for i in range(len(y)):
              a,b = x[i], y[i]
              inp = a              
              inp = transform(a)
              inp = inp.float()
              inp = inp.to(device)    
              outputs = list(flow(inp)[0][0]) 
              print(outputs) 
              c.append(outputs.index(max(outputs))) 
          for i in range(len(c)):
              if (c[i]==y[i]):
                  count += 1
              else:
                  incorrect += 1 
          c_CF = c_CF + c
          y_CF = y_CF + y
          print("Batch",j)
          print("Accuracy="+str(count/(count+incorrect)))
          print("Correctly classified : ",count)
          print("Incorrectly classified", incorrect)  
    print("\n\n\n\n")
    print("Accuracy="+str(count/(count+incorrect)))
    print("Correctly classified : ",count)
    print("Incorrectly classified", incorrect)

    return samples, c_CF, y_CF

# In[13]:


def calculate_test_accuracy():    
    global x
    global y
    global flow
    samples = 0
    count = 0
    incorrect = 0
    c_CF = []
    y_CF = []
    flow.eval()
    tq = get_index(len(val_data))
    confusion_matrix = torch.zeros(2, 2)
    #random.shuffle(val_data)
    print("Calculating test accuracy....") 
    with torch.no_grad():
      for j in tq:
          c = []
          x,y = get_val_data(j)
          y = get_labels(y)
          samples += len(y)
          for i in range(len(y)):
              a,b = x[i], y[i]
              inp = a              
              inp = transform(a)
              inp = inp.float()
              inp = inp.to(device)
              outputs = list(flow(inp)[0][0]) 
              print(outputs)  
              c.append(outputs.index(max(outputs))) 
          for i in range(len(c)):
              if (c[i]==y[i]):
                  count += 1
              else:
                  incorrect += 1 
          c_CF = c_CF + c
          y_CF = y_CF + y
          print("Batch",j)
          print("Accuracy="+str(count/(count+incorrect)))
          print("Correctly classified : ",count)
          print("Incorrectly classified", incorrect)  
    print("\n\n\n\n")
    print("Accuracy="+str(count/(count+incorrect)))
    print("Correctly classified : ",count)
    print("Incorrectly classified", incorrect)

    return samples, c_CF, y_CF

# In[14]:


def confusion_matrix(samples, pred, target):
    nb_classes = 2
    conf_matrix = torch.zeros(nb_classes, nb_classes)
    for t, p in zip(target, pred):
        conf_matrix[t, p] += 1

    print('Confusion matrix\n', conf_matrix)

    TP = conf_matrix.diag()
    for c in range(nb_classes):
        idx = torch.ones(nb_classes).byte()
        idx[c] = 0
        # all non-class samples classified as non-class
        TN = conf_matrix[idx.nonzero()[:, None], idx.nonzero()].sum() #conf_matrix[idx[:, None], idx].sum() - conf_matrix[idx, c].sum()
        # all non-class samples classified as class
        FP = conf_matrix[idx, c].sum()
        # all class samples not classified as class
        FN = conf_matrix[c, idx].sum()
    
        print('Class {}\nTP {}, TN {}, FP {}, FN {}'.format(
            c, TP[c], TN, FP, FN))

        Sensitivity = TP[c]/(TP[c]+FN)
        Specificity = TN/(FP+FN)
        Precision = TP[c]/(TP[c]+FP)
        Recall = TP[c]/(TP[c]+FN)
        F_Measure = (2 * Precision * Recall) / (Precision + Recall)
        print('Sensitivity:', Sensitivity)
        print('Specificity:', Specificity)
        print('Precision:',Precision)
        print('Recall:',Recall)
        print('F-Measure:', F_Measure)

# In[15]:


optimizer = optim.SGD(flow.parameters(), lr=0.0000001, momentum=0.9)
#optimizer = optim.SGD(flow.parameters(), lr=0.0000001, momentum=0.5)

# In[ ]:

offset = 0
asd = []
qw = 0
for epoch in range(200):  # loop over the dataset multiple times
    s1 = time.time()
    tq = get_index(len(train_data)-1)
    random.shuffle(train_data)
    epoch_loss = 0.0
    for j in tq:
      #path = get_path()
      #print(path)
      x,y = get_train_data(j)
      running_loss = 0.0      
      x,y = shuffle(x,y)
      y = get_labels(y)
      #print(f"num of classes:{len(y)}")
      for i in range(len(y)):
          a,b = x[i], y[i]
          optimizer.zero_grad()
          inp = transform(a)
          inp = inp.float()
          inp = inp.to(device)
          label = torch.tensor([b],dtype=torch.long)
          label = label.to(device)
          outputs = flow(inp)
          loss = criterion(outputs[0], label)
          loss.backward()
          optimizer.step()
          running_loss += loss.item()
          epoch_loss += loss.item()
          #print(f"num of mini batches:{q}")
          if i % 3 == 0:            
              print('[%d, %5d] loss: %.3f ' %(epoch + 1, i + 1, running_loss/3))
              t.append(running_loss / 3)
              running_loss = 0.0
              qw += 3
    
    print('[%d] Epoch loss: %.3f ' %(epoch + 1, epoch_loss/qw))
    qw = 0
    asd.append(epoch_loss / 1260)
    s2 = time.time()    
    print(f"Time taken for {epoch + 1} epoch : {s2-s1}\n\n\n ")

    dump("FlowModel.pkl",f"Final_Binary_Classification_Dataset/Flow_Model_28Nov_lre-7/FlowModel_{offset + epoch + 1}.pkl",flow)
    dump_pickle("FlowModelLoss.pkl",f"Final_Binary_Classification_Dataset/Flow_Model_28Nov_lre-7/FlowModel_Loss_{offset + epoch + 1}.pkl",loss)
    #plot_loss(t)
    
    '''if((epoch+1)%4 == 0):
        samples, pred, tar = calculate_train_accuracy()
        confusion_matrix(samples, pred, tar)
        samples, pred, tar = calculate_test_accuracy()
        confusion_matrix(samples, pred, tar)'''
    

plot_loss(t)

print('Finished Training')
