import os
from matplotlib import pyplot as plt
import numpy as np

#pengambilan data feature awal
def getFeature(params):
    coor_x = []
    coor_y = []
    
    new_coor_x = []
    new_coor_y = []
    new_temp_x = []
    new_temp_y = []
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
        
        coor_x.append(int(words[0]))
        coor_y.append(int(words[1]))
        if words[3] == '1' :
            new_temp_x.append(int(words[0]))
            new_temp_y.append(int(words[1])) 
        
        if words[3] == '0' :
            if len(new_temp_x)>0:
                new_coor_x.append(new_temp_x)
                new_coor_y.append(new_temp_y)
                new_temp_x=[]
                new_temp_y=[]
        
    out = [coor_x,coor_y,new_coor_x,new_coor_y,]
    return out

data = getFeature([r"Task2/U29S1.TXT"])
plt.plot(data[0], data[1], color='black')
plt.show()

for i,r in enumerate(data[2]):
    plt.plot(data[2][i], data[3][i], color='black')
plt.show()
