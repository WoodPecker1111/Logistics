from numpy import *


def loadDataSet(filename):   #读取数据（这里只有两个特征）
    dataMat = []
    labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split(',')
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1]),
                        float(lineArr[2]), float(lineArr[3])])
        #前面的1，表示方程的常量。比如四个特征X1,X2,X3,X4，共需要五个参数，W0+W1*X1+W2*X2+W3*X3+W4*X4
        labelMat.append(int(lineArr[4]))

    return dataMat,labelMat

def sigmoid(inX):  #sigmoid函数
    return 1.0/(1+exp(-inX))

def gradAscent(dataMat, labelMat): #梯度上升求最优参数
    dataMatrix=mat(dataMat) #将读取的数据转换为矩阵
    classLabels=mat(labelMat).transpose() #将读取的数据转换为矩阵
    m,n = shape(dataMatrix)
    alpha = 0.001  #设置梯度的阀值，该值越大梯度上升幅度越大
    maxCycles = 500  #设置迭代的次数，一般看实际数据进行设定，有些可能200次就够了
    weights = ones((n,1)) #设置初始的参数，并都赋默认值为1。注意这里权重以矩阵形式表示五个参数。
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)
        error = (classLabels - h)     # 求导后差值
        weights = weights + alpha * dataMatrix.transpose()* error # 迭代更新权重
    return weights

def stocGradAscent0(dataMat, labelMat):  #随机梯度上升，当数据量比较大时，每次迭代都选择全量数据进行计算，计算量会非常大。所以采用每次迭代中一次只选择其中的一行数据进行更新权重。
    dataMatrix=mat(dataMat)
    classLabels=labelMat
    m,n=shape(dataMatrix)
    alpha=0.01
    epoch = 1000
    weights = ones((n,1))
    for k in range(epoch):
        for i in range(m):  # 遍历计算每一行
            h = sigmoid(sum(dataMatrix[i] * weights))
            error = classLabels[i] - h
            weights = weights + alpha * error * dataMatrix[i].transpose()
    return weights

def stocGradAscent1(dataMat, labelMat): #改进版随机梯度上升，在每次迭代中随机选择样本来更新权重，并且随迭代次数增加，权重变化越小。
    dataMatrix=mat(dataMat)
    classLabels=labelMat
    m,n=shape(dataMatrix)
    weights=ones((n,1))
    maxCycles=500
    for j in range(maxCycles): #迭代
        dataIndex=[i for i in range(m)]
        for i in range(m): #随机遍历每一行
            alpha=4/(1+j+i)+0.0001  #随迭代次数增加，权重变化越小。
            randIndex=int(random.uniform(0,len(dataIndex)))  #随机抽样
            h=sigmoid(sum(dataMatrix[randIndex]*weights))
            error=classLabels[randIndex]-h
            weights=weights+alpha*error*dataMatrix[randIndex].transpose()
            del(dataIndex[randIndex]) #去除已经抽取的样本
    return weights


def test(dataMat, labelMat,weights):
    rcount = 0
    for i in range(len(dataMat)):
        h = sigmoid(sum(dataMat[i] * weights))
        if h>0.5:
            label = 1
        else:
            label = 0
        if label is int(labelMat[i]):
            rcount += 1
    rightrate = rcount/len(dataMat)
    errorrate = 1 - rightrate
    return rightrate, errorrate


def main():
    dataMat, labelMat = loadDataSet('data_banknote_authentication_TrainData.txt')

    weights1 = gradAscent(array(dataMat), labelMat)
    weights2 = stocGradAscent0(array(dataMat), labelMat)
    weights3 = stocGradAscent1(array(dataMat), labelMat)
    print("梯度上升得到的权重：",weights1)
    print("随机梯度上升得到的权重：",weights2)
    print("改进梯度上升得到的权重：",weights3)
    testdata, testlabels = loadDataSet('data_banknote_authentication_TestingData.txt')
    rightrate1, errorrate1 = test(testdata, testlabels, weights1)
    rightrate2, errorrate2 = test(testdata, testlabels, weights2)
    rightrate3, errorrate3 = test(testdata, testlabels, weights3)

    print("梯度上升:正确率:",rightrate1, ",错误率:",errorrate1)
    print("随机梯度上升:正确率:",rightrate2, ",错误率:",errorrate2)
    print("改进梯度上升:正确率:",rightrate3, ",错误率:",errorrate3)

if __name__=='__main__':
    main()
