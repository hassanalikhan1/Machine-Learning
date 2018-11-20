
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
        if int(row[0])==0:
            lDict[int(row[1])]=-1
        else:
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



def gradientDescent(completeData,label_data):

    trainDict={}
    testDict={}

    emp_risk=0
    diff=1

    eta=0.001
    for index in range(len(completeData)):
        
        if index in label_data.keys():
            trainDict[index]=(completeData[index])
        else:
            testDict[index]=completeData[index]

  
    vecW=[]
    rows=len(trainDict)
    cols=len(trainDict[1])
    ##creating a vector w containing random numbers from -0.01 to 0.01, perpendicular to the data vector
    for x in range(cols):
        vecW.append(0.02 * random.random()-0.01)

    train_rows=trainDict.values()
    # create an  R vector that contains the output of the training data
    
    
    prevError=None
    
    while (diff>0.001):
        dell=[0] * cols

        for k,v in trainDict.items():
            a=(label_data[k] * dot(vecW,v))
            for j in range(len(v)):
                if a<1:
                    dell[j]+= -(label_data[k]*v[j])

        eta_list = [1, .1, .01, .001, .0001, .00001, .000001, .0000001, .00000001, .000000001, .0000000001, .00000000001]
        bestobj = 1000000000000

        for k in range(0, len(eta_list), 1):
            eta = eta_list[k]
            # update w
            for j in range(0, cols, 1):
                vecW[j] -= eta * dell[j]

            emp_risk = 0

            for k,v in trainDict.items():
                emp_risk+= max(0,1 - (label_data[k] * dot(vecW,v)))
                diff=abs(prev-emp_risk)

            obj = emp_risk
            if obj < bestobj:
                bestobj = obj
                best_eta = eta

                # update w
            for j in range(0, cols, 1):
                vecW[j] += eta * dell[j]

        if best_eta != None:
            eta = best_eta

        for j in range(cols):
            vecW[j] -=eta *dell[j]

        emp_risk = 0
        
        for k,v in trainDict.items():
            emp_risk+= max(0,1 - (label_data[k] * dot(vecW,v)))
            diff=abs(prev-emp_risk)
    

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
    