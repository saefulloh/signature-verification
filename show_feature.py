import os
from matplotlib import pyplot as plt
import numpy as np
import statistics
from scipy.spatial import distance
from numpy.random import uniform
from scipy import interpolate
from scipy.special import eval_hermite
from scipy.signal import hilbert
from scipy.fftpack import dct
from scipy.fftpack import dst
import math
from math import floor
import pandas as pd
from pywt import dwt
from scipy.optimize import brentq
from scipy.interpolate import interp1d


#Ambil jumlah baris per signature getMax([r"Task2/U17S3.TXT"])
def getMax(params): 
    #print(params)
    file1 = open(params[0], 'r')
    maxLinesNew = 0 
    maxLinesRaw = 0
    while True:
        line = file1.readline()  
        if not line: 
            break
        line = line.replace("\n","")
        words = line.split(" ")
        if len(words) == 1 :
            maxLinesRaw = int(words[0])
            continue
            
        if words[3] == '0' :
            continue
            
        maxLinesNew = maxLinesNew + 1
       
    out = [maxLinesNew,maxLinesRaw]
    return out;

#ambil max baris per user getRawFeature([r"Task2/U17S3.TXT","Putus"])
def getMaxValue(u):
    datapath = r"Task2"
    listFiles = os.listdir(datapath)
    listFiles.sort()
    maxLine = [0,0]
    for fileName in listFiles:
        
        sPotition = fileName.find('S')
        number = int(fileName[sPotition+1:len(fileName)-4])
        userNumber = int(fileName[1:sPotition])
        
        if u != userNumber:
            continue
            
        maxLineTemp = getMax([datapath+"/"+fileName])
        if maxLine[0] < maxLineTemp[0] :
            maxLine=maxLineTemp
    return maxLine

#pengambilan data feature awal
def getRawFeature(params):
    signatureType="Putus";
    ind = []
    List_coor = []
    y = []
    no_of_img = 0
    List_coor_target = []
    List_coor_target_y = []
    List_coor_target_azi = []
    List_coor_target_alt = []
    List_coor_target_p = []
    List_coor_target_t = []
    time = 0
    count = 0
    k = 0
    numberOfLine = 0
    file1 = open(params[0], 'r')
    while True:
        line = file1.readline()  
        if not line: 
            break       
        line = line.replace("\n","")
        words = line.split(" ")
        if len(words) == 1 :
            maxLine = int(words[0])
            continue
        if signatureType == "Putus":
            if words[3] == '0' :
                continue
        
        numberOfLine = numberOfLine + 1
        
        #List_coor.append([no_of_img , time , code , code_y , code_azi , code_alt , code_p])
        List_coor.append([no_of_img , numberOfLine , words[0] , words[1] , words[4] , words[5] , words[6]])
        List_coor_target.append(words[0])
        List_coor_target_y.append(words[1])
        List_coor_target_azi.append(words[4])
        List_coor_target_alt.append(words[5])
        List_coor_target_p.append(words[6])
        List_coor_target_t.append(words[2])
        length_target = len(List_coor_target)
    
    out = [List_coor,List_coor_target,List_coor_target_y,List_coor_target_azi,List_coor_target_alt,List_coor_target_p,List_coor_target_t]
    return out

##menghasilkan titik-titik data baru dalam suatu jangkauan dari suatu set diskret data-data yang diketahui.
def interpolation(params):
    #print("params 0",params[2],params[0])
    length_target = len(params[0])
    x = []
    for qwe in range(length_target):
      x.append(qwe)
    y_x = params[0]
    f = interpolate.interp1d(x, y_x)
    size_extra = params[1] - length_target
    sampling = uniform(low=0.0, high=length_target-1, size=size_extra)
    x_new = x
    for qw in range(len(sampling)):
      x_new.append(sampling[qw])
    x_new = np.sort(x_new)
    ynew_x = f(x_new)   # use interpolation function returned by `interp1d`
    #List_coor_target = list(map(int, ynew_x))
    List_coor_target=ynew_x
    #print("params new ",params[2],List_coor_target)
    return List_coor_target

def getFeatureAwal(params):
    maximum = params[1]
    feature = getRawFeature([params[0]])
    List_coor_target = interpolation([feature[1],maximum,'target'])
    List_coor_target_y = interpolation([feature[2],maximum,'target_y'])
    List_coor_target_p = interpolation([feature[5],maximum,'target_p'])
    List_coor_target_alt = interpolation([feature[4],maximum,'target_alt'])
    List_coor_target_azi = interpolation([feature[3],maximum,'target_azi'])
    List_coor_target_t = interpolation([feature[6],maximum,'target_t'])
    #print(List_coor_target_t)
    out = [List_coor_target,List_coor_target_y,List_coor_target_p,List_coor_target_alt,List_coor_target_azi,List_coor_target_t]
    #out = [feature[1],feature[2],feature[5],feature[4],feature[3],,feature[6]]
    return out
    
def getPhysicalFeature(params):
    maximum = params[1]
    feature = getRawFeature([params[0]])
    List_coor_target = interpolation([feature[1],maximum,'target'])
    List_coor_target_y = interpolation([feature[2],maximum,'target_y'])
    List_coor_target_p = interpolation([feature[5],maximum,'target_p'])
    List_coor_target_alt = interpolation([feature[4],maximum,'target_alt'])
    List_coor_target_azi = interpolation([feature[3],maximum,'target_azi'])
    List_coor_target_t = interpolation([feature[6],maximum,'target_t'])

    #Physical features
    disp = []
    velo_x = []
    velo_y = []
    velocity_x = []
    velocity_y = []
    abs_velo = []
    abs_velo = []
    ac_x = []
    ac_y = []
    abs_ac = []
    cent_ac = []
    cos_alpha = []
    sin_alpha = []
    cos_beta = []
    theta = []
    ang_velo = []
    for n in range(len(List_coor_target)):
      disp.append(math.sqrt((List_coor_target[n] **2)+ (List_coor_target_y[n] **2)))
      if(n< len(List_coor_target) - 1):
        velo_x.append((List_coor_target[n+1] - List_coor_target[n]))
        velo_y.append((List_coor_target_y[n+1] - List_coor_target_y[n]))
        if((List_coor_target_t[n+1] - List_coor_target_t[n]) == 0):
          #fai tambah
          if n == 0 :
            velocity_x.append(0)
            velocity_y.append(0)
          else:
            velocity_x.append(velocity_x[n-1])
            velocity_y.append(velocity_y[n-1])
        else:
          velocity_x.append((List_coor_target[n+1] - List_coor_target[n])/(List_coor_target_t[n+1] - List_coor_target_t[n]))
          velocity_y.append((List_coor_target_y[n+1] - List_coor_target_y[n])/(List_coor_target_t[n+1] - List_coor_target_t[n]))
        abs_velo.append(math.sqrt((velocity_x[n] ** 2) + (velocity_y[n] ** 2)))
        if(abs_velo[n] == 0 and n != 0):
          cos_alpha.append(cos_alpha[n-1])
          sin_alpha.append(sin_alpha[n-1])
          cos_beta.append(cos_beta[n-1])
        elif ( n == 0 and abs_velo[n] == 0):
          cos_alpha.append(np.NaN)
          sin_alpha.append(np.NaN)
          cos_beta.append(np.NaN)
        else:
          cos_alpha.append((List_coor_target[n+1] - List_coor_target[n])/abs_velo[n])
          sin_alpha.append((List_coor_target_y[n+1] - List_coor_target_y[n])/abs_velo[n])
          cos_beta.append(velo_x[n] / abs_velo[n])
        if((List_coor_target[n+1] - List_coor_target[n]) != 0):
          theta.append((math.atan((List_coor_target_y[n+1] - List_coor_target_y[n])/(List_coor_target[n+1] - List_coor_target[n]))))
        if((List_coor_target[n+1] - List_coor_target[n]) == 0):
          theta.append(3.145/2)
    for n in range(len(List_coor_target) - 2):
      ang_velo.append((theta[n+1] - theta[n]))
      if(n < len(List_coor_target) - 3):
        if((List_coor_target_t[n+1] - List_coor_target_t[n]) == 0):
            if n == 0 :
              ac_x.append(0)
              ac_y.append(0)
            else:
              ac_x.append(ac_x[n-1])
              ac_y.append(ac_y[n-1])
        else:
          ac_x.append((velo_x[n+1] - velo_x[n])/(List_coor_target_t[n+1] - List_coor_target_t[n]))
          ac_y.append((velo_y[n+1] - velo_y[n])/(List_coor_target_t[n+1] - List_coor_target_t[n]))
        abs_ac.append((math.sqrt((ac_x[n] ** 2) + (ac_y[n] ** 2))))
        if(abs_velo[n] == 0 and n != 0):
          cent_ac.append(cent_ac[n-1])
        elif(abs_velo[n] == 0 and n ==0):
          cent_ac.append(np.NaN)  
        else:
          cent_ac.append(((velo_x[n] * ac_y[n]) - (velo_y[n] * ac_x[n]))/abs_velo[n])
        
    List_coor_target = List_coor_target.tolist()
    List_coor_target_y = List_coor_target_y.tolist()
    List_coor_target_p = List_coor_target_p.tolist()
    feature_set = [disp,velocity_x,velocity_y,abs_velo,cos_alpha,sin_alpha,cos_beta,theta,ang_velo,ac_x,ac_y,abs_ac,cent_ac,List_coor_target,List_coor_target_y,List_coor_target_p]
    out = feature_set
    return out

#maxLine = getMaxFromAll(20)
#print("maxLine",maxLine)
#pf = getPhysicalFeature([r"Task2/U20S1.TXT",maxLine[0]])


def getFrequencyFeature(params):
    maximum = params[1]
    feature = getRawFeature([params[0]])
    List_coor_target = interpolation([feature[1],maximum,'target'])
    List_coor_target_y = interpolation([feature[2],maximum,'target_y'])
    List_coor_target_p = interpolation([feature[5],maximum,'target_p'])
    List_coor_target_alt = interpolation([feature[4],maximum,'target_alt'])
    List_coor_target_azi = interpolation([feature[3],maximum,'target_azi'])
    List_coor_target_t = interpolation([feature[6],maximum,'target_t'])

    dct_x = dct(List_coor_target)
    dct_y = dct(List_coor_target_y)
    dct_p = dct(List_coor_target_p)
    dct_alt = dct(List_coor_target_alt)
    dct_azi = dct(List_coor_target_azi)
    
    dct_x = dct_x.tolist()
    dct_y = dct_y.tolist()
    dct_p = dct_p.tolist()
    dct_alt = dct_alt.tolist()
    dct_azi = dct_azi.tolist()
    
    feature_set = [dct_x, dct_y, dct_p, dct_alt, dct_azi]
    out = feature_set
    return out

def getStatisticalFeature(params):
    maximum = params[1]
    feature = getRawFeature([params[0]])
    List_coor_target = interpolation([feature[1],maximum,'target'])
    List_coor_target_y = interpolation([feature[2],maximum,'target_y'])
    List_coor_target_p = interpolation([feature[5],maximum,'target_p'])
    List_coor_target_alt = interpolation([feature[4],maximum,'target_alt'])
    List_coor_target_azi = interpolation([feature[3],maximum,'target_azi'])
    List_coor_target_t = interpolation([feature[6],maximum,'target_t'])

    result_x = np.correlate(List_coor_target, List_coor_target, mode='full')
    result_x = result_x[floor(result_x.size/2):].tolist()
    result_y = np.correlate(List_coor_target_y, List_coor_target_y, mode='full')
    result_y = result_y[floor(result_y.size/2):].tolist()
    result_p = np.correlate(List_coor_target_p, List_coor_target_p, mode='full')
    result_p = result_p[floor(result_p.size/2):].tolist()
    result_alt = np.correlate(List_coor_target_alt, List_coor_target_alt, mode='full')
    result_alt = result_alt[floor(result_alt.size/2):].tolist()
    result_azi = np.correlate(List_coor_target_azi, List_coor_target_azi, mode='full')
    result_azi = result_azi[floor(result_azi.size/2):].tolist()
    
    feature_set = [result_x, result_y, result_p, result_alt, result_azi]
    out = feature_set
    return out

def visualiseFitur(fileName):
    
    sPotition = fileName.find('S')
    number = int(fileName[sPotition+1:len(fileName)-4])
    userNumber = int(fileName[1:sPotition])
    fileName = r"Task2/"+fileName;
    maxLine = getMaxValue(userNumber)
    feature_awal=getFeatureAwal([fileName,maxLine[0]])
    feature_Physical_set = getPhysicalFeature([fileName,maxLine[0]])
    feature_Frequency_set = getFrequencyFeature([fileName,maxLine[0]])
    feature_Statistical_set = getStatisticalFeature([fileName,maxLine[0]])
    
    plt.plot(feature_awal[0],label="x")
    plt.plot(feature_awal[1],label="y")
    plt.plot(feature_awal[2],label="p")
    plt.plot(feature_awal[3],label="alt")
    plt.plot(feature_awal[4],label="azi")
    #plt.plot(feature_awal[5],label="t")
    plt.legend()
    plt.title("Fitur Awal")
    plt.show()
    
    plt.plot(feature_Physical_set[0],label="disp")
    plt.plot(feature_Physical_set[13],label="x")
    plt.plot(feature_Physical_set[14],label="y")
    plt.plot(feature_Physical_set[15],label="p")
    plt.legend()
    plt.title("Fitur Fisik")
    plt.show()

    plt.plot(feature_Physical_set[1],label="vel_x")
    plt.plot(feature_Physical_set[2],label="vel_y")
    plt.plot(feature_Physical_set[3],label="cos_alpha")
    plt.plot(feature_Physical_set[4],label="sin_alpha")
    plt.plot(feature_Physical_set[5],label="cos_beta")
    plt.plot(feature_Physical_set[6],label="theta")
    plt.plot(feature_Physical_set[7],label="ang_velo")
    plt.plot(feature_Physical_set[8],label="ac_x")
    plt.plot(feature_Physical_set[9],label="ac_y")
    plt.plot(feature_Physical_set[10],label="abs_ac")
    plt.plot(feature_Physical_set[11],label="abs_ac")
    plt.plot(feature_Physical_set[12],label="cent_ac")
    plt.legend()
    plt.title("Fitur Fisik")
    plt.show()

    plt.plot(feature_Frequency_set[0],label="x")
    plt.plot(feature_Frequency_set[1],label="y")
    plt.plot(feature_Frequency_set[2],label="p")
    plt.plot(feature_Frequency_set[3],label="alt")
    plt.plot(feature_Frequency_set[4],label="azi")
    plt.legend()
    plt.title("Fitur Frekuensi")
    plt.show()

    plt.plot(feature_Statistical_set[0],label="x")
    plt.plot(feature_Statistical_set[1],label="y")
    plt.plot(feature_Statistical_set[2],label="p")
    plt.plot(feature_Statistical_set[3],label="alt")
    plt.plot(feature_Statistical_set[4],label="azi")
    plt.legend()
    plt.title("Fitur Statistik")
    plt.show()
