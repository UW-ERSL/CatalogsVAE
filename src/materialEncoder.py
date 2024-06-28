from networks import VariationalAutoencoder
import torch
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon
import numpy as np
from utilFuncs import to_np
import pickle

#--------------------------#
class MaterialEncoder:
  def __init__(self, trainingData, dataInfo, dataIdentifier, vaeSettings):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # self.device = "cpu"
    print(self.device)
    self.trainingData, self.dataInfo = trainingData, dataInfo
    self.dataIdentifier = dataIdentifier
    self.vaeSettings = vaeSettings
    self.vaeNet = VariationalAutoencoder(vaeSettings).to(self.device)
  #--------------------------#
  def loadAutoencoderFromFile(self, fileName):
    with open('./results/vaeTrained.pkl', 'r') as f:
      obj0 = pickle.load(f)
      self.vaeNet.load_state_dict(torch.load(obj0))
      self.vaeNet.encoder.isTraining = False
  #--------------------------#
  def trainAutoencoder(self, numEpochs, klFactor, savedNet, learningRate):
    opt = torch.optim.Adam(self.vaeNet.parameters(), learningRate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    convgHistory = {'reconLoss':[], 'klLoss':[], 'loss':[]}
    self.vaeNet.encoder.isTraining = True
    
    self.trainingData=self.trainingData.to(self.device)
    
    for epoch in range(numEpochs):
      opt.zero_grad()
      predData = self.vaeNet(self.trainingData)
      klLoss = klFactor*self.vaeNet.encoder.kl

      reconLoss =  ((self.trainingData - predData)**2).sum()
      loss = reconLoss + klLoss 
      loss.backward()
      convgHistory['reconLoss'].append(reconLoss)
      convgHistory['klLoss'].append(klLoss/klFactor) # save unscaled loss
      convgHistory['loss'].append(loss)
      opt.step()
      if(epoch%500 == 0):
        print('Iter {:d} reconLoss {:.2E} klLoss {:.2E} loss {:.2E}'.\
              format(epoch, reconLoss.item(), klLoss.item(), loss.item()))
    self.vaeNet.encoder.isTraining = False
    with open('./results/vaeTrained.pkl', 'wb+') as f:
      pickle.dump([self.vaeNet.encoder.state_dict()], f)
    return convgHistory
  #--------------------------#
  def getClosestMaterialFromZ(self, z, numClosest = 1):
    zData = self.vaeNet.encoder.z.to('cpu').detach().numpy()
    dist = np.linalg.norm(zData- to_np(z), axis = 1)
    meanDist = np.max(dist)
    distOrder = np.argsort(dist)
    matToUseFromDB = {'material':[], 'confidence':[]}
    for i in range(numClosest):
      mat = self.dataIdentifier['name'][distOrder[i]]
      matToUseFromDB['material'].append(mat)
      confidence = 100.*(1.- (dist[distOrder[i]]/meanDist))
      matToUseFromDB['confidence'].append(confidence)
      print(f"closest material {i} : {mat} , confidence {confidence:.2F}")
    return matToUseFromDB
  #--------------------------#

  
    
