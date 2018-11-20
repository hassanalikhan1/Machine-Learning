
import random
import numpy as np
import math
import sys
'''Get feature data from file as a matrix with a row per data instance'''
def getFeatureData(featureFile):
    x=[]
    dFile = open(featureFile, 'r')
    for line in dFile:
        row = line.split()
        rVec = [float(item) for item in row]
        x.append(rVec)
    dFile.close()
    return x

'''Get label data from file as a dictionary with key as data instance index
and value as the class index
'''
def getLabelData(labelFile):

    lFile = open(labelFile, 'r')
    lDict = {}
    for line in lFile:
        row = line.split()
        lDict[int(row[1])] = int(row[0])
    lFile.close()
    return lDict

def subtractData(A,B):

    ans=[]

    for i in range(len(A)):
        ans.append(A[i]-B[i])
    return ans

def matmult(A,B):

    diff= []

    for i in range(len(A)):
        sumN=0

        for x in range(len(A[i])):
            sumN=sumN+A[i][x]*B[x]
        diff.append(sumN)
    return diff
#dot product function

def dot_product(arg1,arg2):
        dp1=0
        for j in range(0,len(arg1),1):
                dp1 += arg1[j]*arg2[j]
        return dp1;

def leastSquare(datatoTrain, labelsToTrain):

    eta=0.001;
    theta=0.001;

    num_features=len(datatoTrain[0])
    num_instances=len(datatoTrain)

    trainData=[]
    for j in range(len(datatoTrain)):
        if j in labelsToTrain.keys():
            trainData.append((labelsToTrain[j]))


    vector_r=[]
    for i in range(len(datatoTrain)):
        if i in labelsToTrain.keys():
            vector_r.append(float(labelsToTrain[i]))

    vector_w = [float(0.02*random.random()- 0.01) for i in range(num_features)]

    error=0
    
    while True:
        prevError=error
        diff=[]
        
        error=0
        for i in range (0,num_instances,1):

            if (labelsToTrain.get(i) != None and labelsToTrain.get(i) == 0):

                dp=dot_product(vector_w,datatoTrain[i]);
                error += (labelsToTrain[i] - dp)**2
                for j in range (0,num_features,1):
                    diff.append(float((labelsToTrain[i]-dp)*datatoTrain[i][j]))        #
        
        eta_list = [1, .1, .01, .001, .0001, .00001, .000001, .0000001, .00000001, .000000001, .0000000001, .00000000001 ]
        bestobj = 1000000000000
        
        for k in range(0, len(eta_list), 1):
            eta = eta_list[k]

            #update w
            for j in range (0,num_features,1):
                vector_w[j]=vector_w[j]-eta*diff[j]

            error=0
            
            for i in range (0,num_instances,1):

                if (labelsToTrain.get(i) != None and labelsToTrain.get(i) == 0):

                    dp=dot_product(vector_w,datatoTrain[i]);
                    error += (labelsToTrain[i] - dp)**2
                
            obj=error
               
            if obj < bestobj:
                bestobj = obj
                best_eta = eta

            for j in range (0,num_features,1):
                vector_w[j]=vector_w[j]-eta*diff[j]

        if best_eta != None:
            eta = best_eta
        
        if abs(prevError-error)<=theta:       
            break 
    
    
    #distance from origin calculation
    print("w vector: ")
    normw=0
    for j in range(0,num_features-1,1):
        print(abs(vector_w[j]),)
        normw += vector_w[j]**2
    print()

    normw = math.sqrt(normw)
    d_origin = abs( vector_w[len(vector_w)-1]/normw)
    print("distance from origin: ",d_origin)


    for i in range(0,len(datatoTrain),1):
        if (labelsToTrain.get(i) != None and labelsToTrain.get(i) == 0):
            dp=0
            for j in range(0,len(datatoTrain[0]),1):
                dp+=datatoTrain[i][j]*vector_w[j]
                
            if dp>0:
                print("1,",i)
            else:
                print("0,",i)

def main():

    trainFile=sys.argv[1]
    labelFile=sys.argv[2]

    train=getFeatureData(trainFile)
    labels=getLabelData(labelFile)
    leastSquare(train,labels)

main()
