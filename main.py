'''
Created on 15 de out de 2017

@author: raul
'''
from CreateBaseFromFile import CreateBaseFromFile
from sklearn.model_selection import train_test_split
from PCA import pca, pcaScore
from classificadores import *
if __name__ == '__main__':
    
    baseClimate = CreateBaseFromFile.createFromFile("Bases/climate",[20],[0,1],1," ")
    baseBank = CreateBaseFromFile.createFromFile("Bases/bankNote",[4],[])
    erroKNNCli = 0
    erroNaiveCli = 0
    erroArvoreCli = 0
    
    erroKNNBank = 0
    erroNaiveBank  = 0
    erroArvoreBank  = 0
    
    qt1 = 0
    qt2 = 0
    
    removerAtrC = 18
    removerAtrB = 4
    for i in range(100):
        train_atr, test_atr, train_classes, test_classes = train_test_split(baseClimate.atributos, baseClimate.classes, test_size=0.5, random_state=i) 
        if(qt1==0):
            qt1 = len(test_classes)
        erroKNNCli = classicarKNN(train_atr, train_classes, test_atr, test_classes) + erroKNNCli
        erroNaiveCli = naiveBayes(train_atr, train_classes, test_atr, test_classes) + erroNaiveCli
        erroArvoreCli = arvoreDecisao(train_atr, train_classes, test_atr, test_classes) + erroArvoreCli
        train_atr, test_atr, train_classes, test_classes = train_test_split(baseBank.atributos, baseBank.classes, test_size=0.5, random_state=i) 
        if(qt2==0):
            qt2 = len(test_classes)
        erroKNNBank = classicarKNN(train_atr, train_classes, test_atr, test_classes) + erroKNNBank 
        erroNaiveBank  = naiveBayes(train_atr, train_classes, test_atr, test_classes) + erroNaiveBank 
        erroArvoreBank  = arvoreDecisao(train_atr, train_classes, test_atr, test_classes) + erroArvoreBank
    print("SEM PCA base Climate:\nerro KNN:%s\nerro NaiveBayes:%s\nerro arvore:%s\n"%(erroKNNCli/qt1,erroNaiveCli/qt1,erroArvoreCli/qt1))
    print("SEM PCA base BankNote:\nerro KNN:%s\nerro NaiveBayes:%s\nerro arvore:%s\n"%(erroKNNBank/qt2,erroNaiveBank/qt2,erroArvoreBank/qt2))
    
    for j in range(removerAtrC):
        erroKNNCli = 0
        erroNaiveCli = 0
        erroArvoreCli = 0
        
        erroKNNCli1 = 0
        erroNaiveCli1 = 0
        erroArvoreCli1 = 0
        for i in range(100):
            baseClimatePCA = pca(baseClimate,j)
            baseClimatePCAScore = pcaScore(baseClimate, j)
            train_atr, test_atr, train_classes, test_classes = train_test_split(baseClimatePCA.atributos, baseClimatePCA.classes, test_size=0.5, random_state=i) 
            erroKNNCli = classicarKNN(train_atr, train_classes, test_atr, test_classes) + erroKNNCli
            erroNaiveCli = naiveBayes(train_atr, train_classes, test_atr, test_classes) + erroNaiveCli
            erroArvoreCli = arvoreDecisao(train_atr, train_classes, test_atr, test_classes) + erroArvoreCli
            
            train_atr, test_atr, train_classes, test_classes = train_test_split(baseClimatePCAScore.atributos, baseClimatePCAScore.classes, test_size=0.5, random_state=i)
            erroKNNCli1 = classicarKNN(train_atr, train_classes, test_atr, test_classes) + erroKNNCli1
            erroNaiveCli1 = naiveBayes(train_atr, train_classes, test_atr, test_classes) + erroNaiveCli1
            erroArvoreCli1 = arvoreDecisao(train_atr, train_classes, test_atr, test_classes) + erroArvoreCli1 
        print("COM PCA base Climate - atr:%s:\nerro KNN:%s\nerro NaiveBayes:%s\nerro arvore:%s\n"%(18-j,erroKNNCli/qt1,erroNaiveCli/qt1,erroArvoreCli/qt1))
        print("COM PCA Score base Climate - atr:%s:\nerro KNN:%s\nerro NaiveBayes:%s\nerro arvore:%s\n"%(18-j,erroKNNCli1/qt2,erroNaiveCli1/qt2,erroArvoreCli1/qt2))
    
    print("\n") 
   
    for j in range(removerAtrB):
        erroKNNBank = 0
        eroNaiveBank  = 0
        erroArvoreBank  = 0
        
        erroKNNBank1 = 0
        erroNaiveBank1  = 0
        erroArvoreBank1  = 0
        for i in range(100):
            baseBankPCA = pca(baseBank,j)
            baseBankPCAScore = pcaScore(baseBank,j)
            train_atr, test_atr, train_classes, test_classes = train_test_split(baseBankPCA.atributos, baseBankPCA.classes, test_size=0.5, random_state=i) 
            erroKNNBank = classicarKNN(train_atr, train_classes, test_atr, test_classes) + erroKNNBank 
            erroNaiveBank  = naiveBayes(train_atr, train_classes, test_atr, test_classes) + erroNaiveBank 
            erroArvoreBank  = arvoreDecisao(train_atr, train_classes, test_atr, test_classes) + erroArvoreBank
            train_atr, test_atr, train_classes, test_classes = train_test_split(baseBankPCAScore.atributos, baseBankPCAScore.classes, test_size=0.5, random_state=i) 
            erroKNNBank1 = classicarKNN(train_atr, train_classes, test_atr, test_classes) + erroKNNBank1 
            erroNaiveBank1  = naiveBayes(train_atr, train_classes, test_atr, test_classes) + erroNaiveBank1 
            erroArvoreBank1  = arvoreDecisao(train_atr, train_classes, test_atr, test_classes) + erroArvoreBank1
        print("COM PCA base BankNote - atr:%s:\nerro KNN:%s\nerro NaiveBayes:%s\nerro arvore:%s\n"%(4-j,erroKNNBank/qt2,erroNaiveBank/qt2,erroArvoreBank/qt2))
        print("COM PCA Score base BankNote - atr:%s:\nerro KNN:%s\nerro NaiveBayes:%s\nerro arvore:%s\n"%(4-j,erroKNNBank1/qt2,erroNaiveBank1/qt2,erroArvoreBank1/qt2))
    pass