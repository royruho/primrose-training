# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 12:13:49 2019

@author: royru
"""

import numpy as np
import cv2
from os import listdir   
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.misc import imread
from scipy.misc import imresize

def get_dir_list(base_path):
    dir_list = []
    base_path = base_path
    for first_level in listdir(base_path):
        path = base_path+'\\'+first_level
        for second_level in listdir(path):
            dir_list.append(path+'\\'+second_level)
    return dir_list

class PCA_cls:
    def __init__(self,data):
        self.data = data
        self.u, self.sigma, self.vt = np.linalg.svd(data,full_matrices=False, compute_uv=True)
    def get_svd(self):
        return self.u,self.sigma,self.vt
    def get_image(self,index,vectors): #index of image in u, how many pc vectors
        image = self.u[index,:vectors]@np.diag(self.sigma[:pc_vectors])@self.vt[:pc_vectors]
        return image
#########################################################################################
if __name__ == "__main__": 
    base_path = r'.\faces94'
    image_index = 2 #index of image to decode
    image_height = 200 
    image_width = 200
    pc_vectors = 60 # how many eigen vectors to compute
    dir_list = get_dir_list(base_path) # create list of all directories
    face_list = []
    flatten_list = []
    for directory in dir_list: 
        file_path = directory +'\\'+listdir(directory)[0]
        if file_path[-3:] == 'jpg' :
            face = imread(file_path,mode='L')
            face = imresize(face,(image_height,image_width))
            face_list.append(face)
            flatten_list.append(np.reshape(face,(1,image_height*image_width)))
        else:
            continue

    plt.imshow(face_list[image_index])
    plt.show()
    flatten_array = flatten_list[0]
    for column in flatten_list[1:]:
        flatten_array = np.vstack(((flatten_array,column)))
    pca = PCA(n_components=pc_vectors)
    principalComponents = pca.fit_transform(flatten_array)
    
    column_face = pca.inverse_transform(principalComponents[image_index])
    face_array = np.reshape(column_face,(image_height,image_width))
    plt.imshow(face_array)
    plt.show()
    
##############################################################################
    #PCA#
    
    pc_cls = PCA_cls(flatten_array)
    image_from_class = pc_cls.get_image(image_index,pc_vectors)
    image_from_class = np.reshape(image_from_class,(image_height,image_width))
    plt.imshow(image_from_class)
    plt.show()
    
    empty = pc_cls.sigma[:pc_vectors]@pc_cls.vt[:pc_vectors]
    empty = np.reshape(empty,(image_height,image_width))
    plt.imshow(1-empty)
    plt.show()
    
    U, Sigma, V = pc_cls.get_svd()
    vector = np.reshape(flatten_array[image_index,:],(1,40000))
    proj = vector@V.T
    amit = proj@V
    amit = np.reshape(amit,(image_height,image_width))
    plt.imshow(amit)
    plt.show()

    
#############################################################################
    #PCA for folder with same person images #
    base_path = r'.\faces94'
    image_index = 2 #index of image to decode
    image_height = 200 
    image_width = 200
    pc_vectors = 19 # how many eigen vectors to compute
    dir_index = 90
    dir_list = get_dir_list(base_path) # create list of all directories
    face_list = []
    flatten_list = []
    for file in listdir(dir_list[dir_index]):            
        file_path = dir_list[dir_index] +'\\'+file
        if file_path[-3:] == 'jpg' :
            face = imread(file_path,mode='L')
            face = imresize(face,(image_height,image_width))
            face_list.append(face)
            flatten_list.append(np.reshape(face,(1,image_height*image_width)))
        else:
            continue
    
    plt.imshow(face_list[image_index])
    plt.show()
    flatten_array = flatten_list[0]
    for column in flatten_list[1:]:
        flatten_array = np.vstack(((flatten_array,column)))
    pca = PCA(n_components=pc_vectors)
    principalComponents = pca.fit_transform(flatten_array)
    
    column_face = pca.inverse_transform(principalComponents[image_index])
    face_array = np.reshape(column_face,(image_height,image_width))
    plt.imshow(face_array)
    plt.show()