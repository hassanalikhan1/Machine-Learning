import sys
import random
import math


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

def split(threshold,col,data):
	left = list()
	right = list()

	for row in data:
		if row[col] < threshold:
			left.append(row)
		else:
			right.append(row)
	return [left, right]  # returns the 2 group


def gini_value(groups, classes):
    left = groups[0]
    right = groups[1]

    tot_rows = len(left) + len(right)
    gini = 0.0
    for group in groups:
        size = len(group)
        if size == 0:
            continue
        prob = 1
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            prob = prob * p
        gini += (prob) * (size / tot_rows)
    return gini


def cartAlgo(trainData,labelData):

	##separate training and prediction data
	predictionary=list()
	nCol=0
	nRow=0
	nValue=0
	nGini=1
	nGroups=None
	simCount=0

	for r in range(len(trainData)):
		if (labelData.get(r) != None):
			trainData[r].append(labelData[r])
		else:
			predictionary.append(trainData[r])

	train = list()
	for r in trainData:
		length = len(r)
		if length == len(trainData[0]):
			train.append(r)
	
	cols=len(train[0])-1
	rows=len(train)
	
	classes=[0,1]


	for col in range(cols):
		for row in range(rows):
			groups=split(train[row][col],col,train)
			gini = gini_value(groups, classes)

			if gini<nGini:
				nCol=col
				nRow=row
				nValue=train[row][col]
				nGini=gini
				nGroups=groups
			elif gini==nGini:
				simCount=simCount+1

	if (simCount==((len(train)*2)-1)):
		nCol=0

		nRowVal=train[0][nCol]
		nRow=0

		for row in range(rows):
			if train[row][nCol]>nRowVal:
				nRow=row
				nRowVal=train[row][nCol]

		nValue=train[nRow][nCol]
		nGini=gini
		nGroups=split(train[nRow][nCol],nCol,train)

	## get the split values

	win_col=list()
	maxNum=-10000

	for r in range(rows):
		win_col.append(train[r][nCol])
	win_col.sort()
	for r in range(rows):
		val=train[r][nCol]
		if val <nValue:
			if val > maxNum:
				maxNum=val
	s= (maxNum+nValue)/2

	return [nCol, nRow,nValue,nGroups,nGini,s]

def main():

    train=sys.argv[1]
    label=sys.argv[2]

    trainingData=getFeatureData(train)
    labels=getLabelData(label)
    
    result=cartAlgo(trainingData,labels)
    print('Best column:', result[0])
    print('Gini Value:', result[4])
    print('Split point value:', result[5])


main()