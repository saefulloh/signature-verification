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
from sklearn.metrics import roc_curve

from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

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
    feature_set = disp + velocity_x + velocity_y + abs_velo + cos_alpha + sin_alpha + cos_beta + theta + ang_velo + ac_x + ac_y + abs_ac + cent_ac + List_coor_target + List_coor_target_y + List_coor_target_p
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
    
    feature_set = dct_x + dct_y + dct_p + dct_alt + dct_azi
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
    
    feature_set = result_x + result_y + result_p + result_alt + result_azi
    out = feature_set
    return out

#[isTesting,indexTesting,jumlah_signature]
def getSampleData(params):
    datapath = r"Task2"
    listFiles = os.listdir(datapath)
    listFiles.sort()
    #print(listFiles)

    file_names=[]
    dataPhysicalSet=[]
    dataFrequencyFeature=[]
    dataStatisticalFeature=[]
    y = []
    ind = [] #index no image
    count = 0
    no_of_img =0

    testing=params[1]
    
    maxLine = getMaxValue(params[2])

    for fileName in listFiles:
        count = count + 1

        sPotition = fileName.find('S')
        number = int(fileName[sPotition+1:len(fileName)-4])
        userNumber = int(fileName[1:sPotition])
        
        if params[2] != userNumber:
            continue
            
        isTesting = number in testing
        if isTesting != params[0] :
            continue

        if(number >20):
          y.append("False")
        else:
          y.append("True")

        no_of_img = no_of_img+1
        ind.append(no_of_img)
        feature_Physical_set = getPhysicalFeature([r""+datapath+"/"+fileName,maxLine[0]])
        feature_Frequency_set = getFrequencyFeature([r""+datapath+"/"+fileName,maxLine[0]])
        feature_Statistical_set = getStatisticalFeature([r""+datapath+"/"+fileName,maxLine[0]])
        
        file_names.append(fileName)
        dataPhysicalSet.append(feature_Physical_set)
        dataFrequencyFeature.append(feature_Frequency_set)
        dataStatisticalFeature.append(feature_Statistical_set)
        #print(no_of_img,fileName,len(feature_set))
    dataPhysicalSet=pd.DataFrame(dataPhysicalSet)
    dataFrequencyFeature=pd.DataFrame(dataFrequencyFeature)
    dataStatisticalFeature=pd.DataFrame(dataStatisticalFeature)
    
    dataPhysicalSet = dataPhysicalSet.interpolate(method = 'linear', axis = 1 , limit_direction = 'both')
    dataPhysicalSet.isnull().values.any()
    
    dataFrequencyFeature = dataFrequencyFeature.interpolate(method = 'linear', axis = 1 , limit_direction = 'both')
    dataFrequencyFeature.isnull().values.any()
    
    dataStatisticalFeature = dataStatisticalFeature.interpolate(method = 'linear', axis = 1 , limit_direction = 'both')
    dataStatisticalFeature.isnull().values.any()
    
    y = pd.Series(y , index = ind)
    return [dataPhysicalSet,dataFrequencyFeature,dataStatisticalFeature,y,file_names]

def runPrediction(user,methods,x_train,y_train,x_test,y_test):

    out = []
    
    if "SVC" in methods:
        #print("SVC")
        clf=SVC(kernel='linear',probability=True)
        clf.fit(x_train,y_train)
        a = clf.predict(x_test)
        b = clf.predict_proba(x_test)
        c = clf.score(x_test,y_test)
        out.append([user,"SVC",a,b,c])

    if "GaussianNB" in methods:
        #print("GaussianNB")
        clf=GaussianNB()
        clf.fit(x_train,y_train)
        a=clf.predict(x_test)
        b = clf.predict_proba(x_test)
        c = clf.score(x_test,y_test)
        out.append([user,"GaussianNB",a,b,c])

    if "BernoulliNB" in methods:
        #print("BernoulliNB")
        clf=BernoulliNB()
        clf.fit(x_train,y_train)
        a=clf.predict(x_test)
        b = clf.predict_proba(x_test)
        c = clf.score(x_test,y_test)
        out.append([user,"BernoulliNB",a,b,c])

    if "DecisionTreeClassifier" in methods:
        #print("DecisionTreeClassifier")
        clf=DecisionTreeClassifier(random_state=0)
        clf.fit(x_train,y_train)
        a=clf.predict(x_test)
        b = clf.predict_proba(x_test)
        c = clf.score(x_test,y_test)
        out.append([user,"DecisionTreeClassifier",a,b,c])
    
    if "AdaBoostClassifier" in methods:
        #print("AdaBoostClassifier")
        clf=AdaBoostClassifier(n_estimators=100,random_state=0)
        clf.fit(x_train,y_train)
        a=clf.predict(x_test)
        b = clf.predict_proba(x_test)
        c = clf.score(x_test,y_test)
        out.append([user,"AdaBoostClassifier",a,b,c])

    if "GradientBoostingClassifier" in methods:
        #print("GradientBoostingClassifier")
        clf=GradientBoostingClassifier(random_state=0)
        clf.fit(x_train,y_train)
        a=clf.predict(x_test)
        b = clf.predict_proba(x_test)
        c = clf.score(x_test,y_test)
        out.append([user,"GradientBoostingClassifier",a,b,c])

    if "RandomForestClassifier" in methods:
        #print("RandomForestClassifier")
        clf=RandomForestClassifier(max_depth=9,random_state=0)
        clf.fit(x_train,y_train)
        a = clf.predict(x_test)
        b = clf.predict_proba(x_test)
        c = clf.score(x_test,y_test)
        out.append([user,"RandomForestClassifier",a,b,c])
    return out

def signVerification(user,signTesting):
    key = 'True' if signTesting < 21 else 'False'
    methods=["SVC","GaussianNB","BernoulliNB","DecisionTreeClassifier","AdaBoostClassifier","GradientBoostingClassifier","RandomForestClassifier"]
    indexTesting = [signTesting]
    dataTrain = getSampleData([False,indexTesting,user])
    x_train_pysical = dataTrain[0]
    x_train_frequency = dataTrain[1]
    x_train_statistical = dataTrain[2]
    y_train = dataTrain[3]
    print("train",user,signTesting,len(dataTrain[0]))
    
    dataTest = getSampleData([True,indexTesting,user])
    x_test_pysical = dataTest[0]
    x_test_frequency = dataTest[1]
    x_test_statistical = dataTest[2]
    y_test = dataTest[3]
    print("test",user,signTesting,len(dataTest[0]),dataTest[4])
    
    res_pysical = runPrediction(user,methods,x_train_pysical,y_train,x_test_pysical,y_test)    
    res_frequency = runPrediction(user,methods,x_train_frequency,y_train,x_test_frequency,y_test)
    res_statistical = runPrediction(user,methods,x_train_statistical,y_train,x_test_statistical,y_test)
    
    out_user_feature_method = []
    out_user_feature = []
    out_user = []
    
    acc_user = [[],[]]
    scoreAll=0
    ansemble2=[]
    
    #proses pysical
    score=0
    acc_feature = [[],[]]
    for i in res_pysical:
        acc_feature[0].append(i[3][0][0])
        acc_feature[1].append(i[3][0][1])
        
        if i[3][0][0] < i[3][0][1]:
            acc = i[3][0][1]
        else:
            acc = i[3][0][0]
        out_user_feature_method.append([user,signTesting,"pysical",i[1],i[3][0][0],i[3][0][1],acc,i[2][0],i[4]])
        score = score+i[4]
        scoreAll = scoreAll+i[4]
    
    accf = sum(acc_feature[0])/len(acc_feature[0])
    acct = sum(acc_feature[1])/len(acc_feature[1])
    predict = 'True' if accf < acct else 'False'
    ansemble1 = 1 if predict == key else 0
    out_user_feature.append([user,signTesting,"pysical",accf,acct,predict,score,ansemble1])
    acc_user[0].append(accf)
    acc_user[1].append(acct)
    if predict == 'True':
        ansemble2.append(1)
    else:
        ansemble2.append(0)
    
    #proses frequency
    score=0
    acc_feature = [[],[]]
    for i in res_frequency:
        acc_feature[0].append(i[3][0][0])
        acc_feature[1].append(i[3][0][1])
        
        if i[3][0][0] < i[3][0][1]:
            acc = i[3][0][1]
        else:
            acc = i[3][0][0]
        out_user_feature_method.append([user,signTesting,"frequency",i[1],i[3][0][0],i[3][0][1],acc,i[2][0],i[4]])
        score = score+i[4]
        scoreAll = scoreAll+i[4]
        
    accf = sum(acc_feature[0])/len(acc_feature[0])
    acct = sum(acc_feature[1])/len(acc_feature[1])
    predict = 'True' if accf < acct else 'False'
    ansemble1 = 1 if predict == key else 0
    out_user_feature.append([user,signTesting,"frequency",accf,acct,predict,score,ansemble1])
    acc_user[0].append(accf)
    acc_user[0].append(acct)
    if predict == 'True':
        ansemble2.append(1)
    else:
        ansemble2.append(0)
    
    #proses statistical
    score=0
    acc_feature = [[],[]]
    for i in res_statistical:
        acc_feature[0].append(i[3][0][0])
        acc_feature[1].append(i[3][0][1])
        
        if i[3][0][0] < i[3][0][1]:
            acc = i[3][0][1]
        else:
            acc = i[3][0][0]
        out_user_feature_method.append([user,signTesting,"statistical",i[1],i[3][0][0],i[3][0][1],acc,i[2][0],i[4]])
        score = score+i[4]
        scoreAll = scoreAll+i[4]
        
    accf = sum(acc_feature[0])/len(acc_feature[0])
    acct = sum(acc_feature[1])/len(acc_feature[1])
    predict = 'True' if accf < acct else 'False'
    ansemble1 = 1 if predict == key else 0
    out_user_feature.append([user,signTesting,"statistical",accf,acct,predict,score,ansemble1])
    acc_user[0].append(accf)
    acc_user[0].append(acct)
    if predict == 'True':
        ansemble2.append(1)
    else:
        ansemble2.append(0)
    
    accf = sum(acc_user[0])/len(acc_user[0])
    acct = sum(acc_user[1])/len(acc_user[1])
    
    acct_ansemble2=sum(ansemble2)/len(ansemble2)
    predict = 'True' if acct_ansemble2 >= 0.5 else 'False'
    if predict == 'False':
        acct_ansemble2 = 1 - acct_ansemble2
    score_ansemble2 =  1 if predict == key else 0
    out_user.append([user,signTesting,accf,acct,scoreAll,predict,acct_ansemble2,score_ansemble2])
        
    return [out_user_feature_method,out_user_feature,out_user]
  
# proses pembelajaran
listUser=range(1,41)
noTesting = [11,12,13,14,15,21,22,23,24,25] 

out_user_feature_method = []
out_user_feature = []
out_user = []

sum_out_user = []
for i in listUser:
    temp_out_user_feature_method = []
    temp_out_user_feature = []
    temp_out_user = [[],[],[]]
    for j in noTesting: 
        rec = signVerification(i,j)
        out_user_feature_method=out_user_feature_method+rec[0]
        out_user_feature=out_user_feature+rec[1]
        out_user=out_user+rec[2]
        
        #temp_out_user_feature_method=out_user_feature_method+rec[0]
        #temp_out_user_feature=out_user_feature+rec[1]
        for r2 in rec[2]:
            temp_out_user[0].append(r2[4])
            temp_out_user[1].append(r2[6])
            temp_out_user[2].append(r2[7])
    sum_out_user.append([i,j,(sum(temp_out_user[0])/len(temp_out_user[0])),(sum(temp_out_user[1])/len(temp_out_user[1])),(sum(temp_out_user[2])/len(temp_out_user[2]))])


#export ke CSV
#out_user_feature_method,out_user_feature,out_user
#user,no,feature,metode,accF,accT,acc,predict,score
#user,no,feature,accF,accT,acc,predict,score
#user,no,accF,accT,acc,predict,score
np.savetxt('out_user_feature_method.csv', out_user_feature_method, fmt='%s', delimiter=',')
np.savetxt('out_user_feature.csv', out_user_feature, fmt='%s', delimiter=',')
np.savetxt('out_user.csv', out_user, fmt='%s', delimiter=',')
np.savetxt('sum_out_user.csv', sum_out_user, fmt='%s', delimiter=',')
