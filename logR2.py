
'''Get feature data from file as a matrix with a row per data instance'''
# coding: utf-8
import math
import sys
import random


def getFeatureData(featureFile):
    x=[]
    dFile = open(featureFile, 'r')
    for line in dFile:
        row = line.split()
        rVec = [float(item) for item in row]
        rVec.append(1)
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

def dot(a, b):

    list1 = []
    for x, y in zip(a, b):
        mul = x * y
        list1.append(mul)
    return sum(list1)

def sigm(a1, b1):
    dp = dot(a1, b1)
    sigmoid = 1 / (1 + math.exp(-1 * dp))
    if (sigmoid >= 1):
        sigmoid = 0.999999

    return sigmoid

def gradientDescent(completeData,label_data):

    trainDict={}
    testDict={}

    emp_risk=0
    diff=1

    eta=0.01
    for index in range(len(completeData)):
        
        if index in label_data.keys():
            trainDict[index]=(completeData[index])
        else:
            testDict[index]=completeData[index]

    ##divide the training set into three partitions
  
    vecW=[]
    rows=len(trainDict)
    cols=len(trainDict[1])
    ##creating a vector w containing random numbers from -0.01 to 0.01, perpendicular to the data vector
    for x in range(cols):
        vecW.append(0.02 * random.random()-0.01)

    train_rows=trainDict.values()
    # create an  R vector that contains the output of the training data
    
    lambdaVec=[0 0.25 0.5 1.0 1.5 2.0 5.0]
    
    prevError=None
    
    while (diff>0.0000001):
        dell=[0] * cols
        for k,v in trainDict.items():
            a=sigm(vecW,v)-label_data[k]
            for j in range(len(v)):
            	dell[j]+= a* v[j]

        for j in range(cols):
            vecW[j] -=eta *dell[j]

        prev=emp_risk
        emp_risk = 0

        for k,v in trainDict.items():
            emp_risk+= -1 * (label_data[k] * math.log(sigm(vecW, v)) + ((1 - label_data[k]) * math.log(1 - sigm(vecW, v))))

            diff=abs(prev-emp_risk)
            
        print (emp_risk)

    print ('W vector is=')
    print(vecW)

    normw = 0
    for i in range(0, len(vecW)-1):
        normw += vecW[i] ** 2


    normw = math.sqrt(normw)

    dist = abs(vecW[len(vecW) - 1] / normw)

    
    print ('Distance from origin is:')
    print(dist)

    preditedL={}

    for k,v in testDict.items():
        dp = dot(vecW, v)
        if (dp > 0):
            preditedL[k]=1
        else:
            preditedL[k]=0

    print (preditedL)



def main():

    train=sys.argv[1]
    label=sys.argv[2]
    completeData=getFeatureData(train)
    trainingLabels=getLabelData(label)
    

    gradientDescent(completeData,trainingLabels)

main()
    