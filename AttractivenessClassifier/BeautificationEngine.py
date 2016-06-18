#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import time
import argparse
import math
import random
import numpy as np

import cv2
import itertools
import os
from os import listdir
from os.path import isfile, join
import copy

from sklearn.svm import SVC
from operator import sub

import pickle #save and read parameters with SVM 

np.set_printoptions(precision=2)

import openface

from scipy.spatial import Delaunay

class BE:
    def __init__ (self, align, net, mypath, imgDim, american, korean, plastic):        
        self.align = copy.copy(align)
        self.net = copy.copy(net)
        self.mypath = copy.copy(mypath)
        self.imgDim = copy.copy(imgDim)
        self.american = copy.copy(american)
        self.korean = copy.copy(korean)
        self.plastic = copy.copy(plastic)
        self.image_edges = []
        self.triangle_list = []
        
    def getlandmark(self,imgPath):

        bgrImg = cv2.imread(imgPath)
        if bgrImg is None:
            raise Exception("Unable to load image: {}".format(imgPath))
        rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

        
        #start = time.time()
        bb = self.align.getLargestFaceBoundingBox(rgbImg)
        if bb is None:
            raise Exception("Unable to find a face: {}".format(imgPath))

        #start = time.time()
        alignedFace = self.align.align(self.imgDim, rgbImg, bb,
                                  landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            raise Exception("Unable to align image: {}".format(imgPath))

        landmarks = self.align.findLandmarks(rgbImg, bb)
        #rgbImg1 = rgbImg
        #print('landmarks count:',len(landmarks))
        #height, width, _ = rgbImg.shape
        #image_edges = np.zeros((len(landmarks),len(landmarks)))

        return landmarks

    def getedge(self,t):
        #存每個"邊"，image_edges[landmark的編號][landmark的編號] = 1(有邊)=0(沒邊)，因為邊是沒有方向的，所以每個方向都存一次
        image_edges = np.zeros((t,t))
        for i in self.triangle_list:
            image_edges[i[0],i[1]]=1
            image_edges[i[1],i[0]]=1
            image_edges[i[1],i[2]]=1
            image_edges[i[2],i[1]]=1
            image_edges[i[2],i[0]]=1
            image_edges[i[0],i[2]]=1
        self.image_edges = image_edges

    def getarea(self,landmarks):#計算某張臉的面積
        area = 0
        #print triangular_list
        for i in self.triangle_list:
            area += 0.5*np.linalg.norm(np.cross(map(sub,list(landmarks[i[1]]),list(landmarks[i[0]])),
                                                map(sub,list(landmarks[i[2]]),list(landmarks[i[0]]))))
        return area

    def getdistancevector(self,landmarks,rgbImg1):#得到normalized distance vector
        rgb = cv2.imread(rgbImg1)
        edgesvector = []
        #t =0
        for i in range(len(self.image_edges)):
            j = 0
            while j < i:
                if self.image_edges[i][j]==1:
                    cv2.line(rgb,landmarks[i],landmarks[j],(125,0,0))
                    edgesvector.append(pow(pow(landmarks[i][0]-landmarks[j][0],2)+pow(landmarks[i][1]-landmarks[j][1],2),0.5))
                    #t += pow(pow(landmarks[i][0]-landmarks[j][0],2)+pow(landmarks[i][1]-landmarks[j][1],2),0.5)
                j+=1
        #cv2.imwrite('test'+str(rgbImg1.split('-')[1].split('.')[0])+'.jpg',rgb)
        #print('edgesvector count:',len(edgesvector))
        t = self.getarea(landmarks)
        return  [i / pow(t,0.5) for i in edgesvector] #每次都除以面積開根號 for normalize
        #return edgesvector

    def get_triangle_list_from_data(self,path):#reload data and get landmark with triangulation 
        o = open(path)
        str = o.readline()
        ap = []
        while str:
            a = []
            a.append(int(str.split(' ')[0]))
            a.append(int(str.split(' ')[1]))
            a.append(int(str.split(' ')[2].split('\n')[0]))
            ap.append(a)
            str = o.readline()
        #print 'triangular count:',len(ap)
        self.triangle_list = ap

    def get_image_data(self,start,end):#get training data
        vector = []
        label = []
        
        for i in range(len(self.american)):
            if i<start or i>end:#for cross-validation
                lm = self.getlandmark(self.mypath+'/american/'+self.american[i])
                vector.append(self.getdistancevector(lm,self.mypath+'/american/'+self.american[i]))
                label.append(self.american[i].split('-')[0])
        for i in range(len(self.korean)):
            if i<start or i>end:
                lm = self.getlandmark(self.mypath+'/korean/'+self.korean[i])
                vector.append(self.getdistancevector(lm,self.mypath+'/korean/'+self.korean[i]))
                label.append(self.korean[i].split('-')[0])

        for i in range(len(self.plastic)):
            if i<start or i>end:
                lm = self.getlandmark(self.mypath+'/plastic/'+self.plastic[i])
                vector.append(self.getdistancevector(lm,self.mypath+'/plastic/'+self.plastic[i]))
                label.append(self.plastic[i].split('-')[0]) 

        return vector, label

    def svm(self, cl, vector, label):#trainning
        cl.fit(vector, label)    
        return cl

    def saveModel(self,clf):#存svm的參數
        s = pickle.dumps(clf)
        f = open("../Models/svm_para","w")
        f.write(s)
        f.close()

    def loadModel(self):#load svm的參數
        f = open("../Models/svm_para")
        stra = f.read()
        clf2 = pickle.loads(stra)
        return clf2

    def test_perImg(self,cl,imagepath):#測試image的分類
        #image = cv2.imread(imagepath)
        lm = self.getlandmark(imagepath)
        vectort = self.getdistancevector(lm,imagepath)
        return cl.predict([vectort])

    def test(self,cl,start,end):#對所有影像的測試分類正確性，對交叉比對或是對training data進行測試
        t=0
        total = 0
        for i in self.american[start:end]:
           total+=1
           if int(self.test_perImg(cl,self.mypath+'/american/'+i)[0])==0:
                t+=1
        for i in self.korean[start:end]:
            total+=1
            if int(self.test_perImg(cl,self.mypath+'/korean/'+i)[0])==1:
                t+=1
        for i in self.plastic[start:end]:
            total+=1
            if int(self.test_perImg(cl,self.mypath+'/plastic/'+i)[0])==2:
                t+=1
        return t,total

    def knn(self,cl,imagepath,k):
        f = open(imagepath)
        image_edg = getedge(68)
        lm = getlandmark(imagepath)
        vectort = getdistancevector(lm,image_edg,imagepath)
        w = {}
        distance_v = {}
        t = 0
        for i in range(len(self.american)):
            if i<start or i>end:
                lm = getlandmark(self.mypath+'/american/'+self.american[i])
                distance_v[str(t)] = getdistancevector(lm,image_edg,self.mypath+'/american/'+self.american[i])
                #print cl.predict([distance_v[str(t)]])[0]
                pred = 0.2
                if cl.predict([distance_v[str(t)]])[0]==1 : pred=1
                sub_distance = np.abs(map(sub,vectort,distance_v[str(t)]))
                if min(sub_distance)==0: w[str(t)] = 0
                else: w[str(t)] = np.divide(pred,np.array(sub_distance))
                t+=1
        for i in range(len(self.korean)):
            if i<start or i>end:
                lm = getlandmark(self.mypath+'/korean/'+self.korean[i])
                pred = 0.2
                distance_v[str(t)] = getdistancevector(lm,image_edg,self.mypath+'/korean/'+self.korean[i])
                if cl.predict([distance_v[str(t)]])[0]==1 : pred=1
                sub_distance = np.abs(map(sub,vectort,distance_v[str(t)]))
                if min(sub_distance)==0: w[str(t)] = 0
                else: w[str(t)] = np.divide(pred,np.array(sub_distance))
                t+=1
        for i in range(len(self.plastic)):
            if i<start or i>end:
                lm = getlandmark(self.mypath+'/plastic/'+self.plastic[i])
                pred = 0.2
                distance_v[str(t)] = getdistancevector(lm,image_edg,self.mypath+'/plastic/'+self.plastic[i])
                if cl.predict([distance_v[str(t)]])[0]==1 : pred=1
                sub_distance = np.abs(map(sub,vectort,distance_v[str(t)]))
                if min(sub_distance)==0: w[str(t)] = 0
                else: w[str(t)] = np.divide(pred,np.array(sub_distance))
                t+=1
        sort_w_index = sorted(w)
        sum_w = 0
        sum_wv = 0
        for i in range(k):
            sum_wv += np.array(w[sort_w_index[len(sort_w_index)-i-1]])*np.array(distance_v[sort_w_index[len(sort_w_index)-i-1]])
            sum_w += np.array(w[sort_w_index[len(sort_w_index)-i-1]])
        return 1.0*sum_wv/sum_w


    def get_image_data_for_changing_distance(self,clf,image_edg):#get trainning data
        for i in range(len(self.american)):
            change_vector = knn(clf,self.mypath+'/american/'+self.american[i],5)
            #print change_vector
            print clf.predict([change_vector])
        for i in range(len(self.plastic)):
            change_vector = knn(clf,self.mypath+'/plastic/'+self.plastic[i],5)
            print clf.predict([change_vector])