'''Get feature data from file as a matrix with a row per data instance'''
# coding: utf-8
import math
import sys

#<~~~~~~~~~~~~~~~~~~~~~~DATA PREPROCESSING~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>


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


#<~~~~~~~~~~~~~~~~~~~~~~NAIVE BAYES~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>

def mean(numbers): 
	
	mN= sum(numbers)/(len(numbers))
	if mN==0:
		mn=1
	return mN


def stdev(numbers):
	meanN=mean(numbers)
	var=sum([pow(i-meanN,2) for i in numbers])/float(len(numbers)-1)
	sd=math.sqrt(var)

	if sd==0:
		sd=1
	return sd
  

def naiveBayesian(completeData,trainLabels):
    
    #separting the training and the testing data

    trainDict={}
    testRows={}
    
    trainDict[0]=[]
    trainDict[1]=[]

    for index in range(len(completeData)):
        if index in trainLabels.keys():
            if trainLabels[index]==0:
                trainDict[0].append(completeData[index])
            else:
                trainDict[1].append(completeData[index])
        else:
            testRows[index]=completeData[index]

    #summarizing the training data by class
    

    summarizeDict={}
    summarizeDict[0]=[]
    summarizeDict[1]=[]
   
    for labelVal,attributes in trainDict.items():
        for attribute in zip(*attributes):
            summarizeDict[labelVal].append([mean(attribute),stdev(attribute)])

    #calculate the probabilties of testing data in each class (if x is closest to the mean of class j normalized by standard deviation)
    testResults={}

    for rowNo,row in testRows.items():

        classProbs={}
        classProbs[0]=0
        classProbs[1]=0

        for classVal, statsDataClass in summarizeDict.items():

            sumProb=0

            for x in range(len(row)):

                sumProb=sumProb+pow((row[x]-statsDataClass[x][0])/(statsDataClass[x][1]),2)
                classProbs[classVal]=sumProb

        if classProbs[0]<classProbs[1]:
            testResults[rowNo]=0
        else:
            testResults[rowNo]=1
    
    #now comparing the test results to the actual labels to find accuracy
    
    print (testResults)

    # hits=0
    # misses=0
    
    # for testRowNo,testLabel in testResults.iteritems():
        
    #     if testRowNo in actualLabels.keys():
            
    #         if testLabel==actualLabels[testRowNo]:
    #             hits=hits+1;
    #         else:
    #             misses=misses+1;


    # print "Hits are: %s"%hits
    # print "Misses are: %s"%misses

    # accuracy= hits/(hits+misses)
    # print accuracy


#<~~~~~~~~~~~~~~~~~~~~~~MAIN FUNCTION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~>

#The main file takes three input strings on command line as arguments : The data containing the features, the training labels and the file containing the actual labels.

def main():
    dataFile=sys.argv[1]
    labelFile=sys.argv[2]

    completeData=getFeatureData(dataFile)
    trainingLabels=getLabelData(labelFile)
    naiveBayesian(completeData,trainingLabels);

main()