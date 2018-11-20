
'''Get feature data from file as a matrix with a row per data instance'''
# coding: utf-8
import math
import sys
import random
import numpy as np 


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
        lDict[int(row[1])]=int(row[0])
    lFile.close()
    return lDict

def subtractMat (A, B):
    
    result=[]
    
    for x in range(len(A)):
        result.append(A[x]-B[x])

    return result


def gradientDescent(completeData,label_data):

    trainDict={}
    testDict={}

    theta=0.001
    eta=0.001
    for index in range(len(completeData)):
        
        if index in label_data.keys():
            trainDict[index]=(completeData[index])
        else:
            testDict[index]=completeData[index]

    dimT=len(trainDict[1])
    vecW=[]

    ##creating a vector w containing random numbers from -0.01 to 0.01, perpendicular to the data vector
    
    for x in range(dimT):
        vecW.append(random.uniform(-0.01,0.01))

    train_rows=trainDict.values()
    # create an  R vector that contains the output of the training data
    
    rVec=[float(label_data[key]) for key,val in trainDict.items()]

    prevError=None
    
    while True:
        
        o_diff=[]
        for ro in train_rows:

        	o_diff.append(np.dot(ro,vecW))

        dVec= subtractMat(rVec,o_diff)    ##compute the differenc vector r-Xw

        error=sum([dVec[i]*dVec[i] for i in range(len(dVec))])   ##compute the error
        print (error)
        
        devMat=eta*(np.dot(dVec,train_rows))

        if prevError is not None and abs(error-prevError) <= theta:
            break


        vecW=vecW+devMat
        prevError=error
    
    print dVec

    print ('W vector is=')
    print(vecW)
    
    hyperplaneDist=vecW[len(vecW)-1]/sum([vecW[i]*vecW[i] for i in range(len(vecW))])
    
    print ('Distance from origin is:')
    print(hyperplaneDist)

    preditedL=[]

def main():

    train=sys.argv[1]
    label=sys.argv[2]
    completeData=getFeatureData(train)
    trainingLabels=getLabelData(label)

    gradientDescent(completeData,trainingLabels)

main()
    