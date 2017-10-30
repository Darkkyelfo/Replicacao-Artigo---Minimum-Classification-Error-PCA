'''
Created on 15 de out de 2017

@author: raul
'''
from CreateBaseFromFile import CreateBaseFromFile
from sklearn.model_selection import train_test_split
from PCA import PCA as PCAR
from Base import Base
from Grafico import GerarGrafico
from sklearn.decomposition import PCA
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
    
    removerAtrC = 18
    removerAtrB = 4
    for i in range(100):
        train_atr, test_atr, train_classes, test_classes = train_test_split(baseClimate.atributos, baseClimate.classes, test_size=0.5, random_state=i) 
        qt1 = len(test_classes)
        erroKNNCli = classicarKNN(train_atr, train_classes, test_atr, test_classes) + erroKNNCli
        erroNaiveCli = naiveBayes(train_atr, train_classes, test_atr, test_classes) + erroNaiveCli
        erroArvoreCli = arvoreDecisao(train_atr, train_classes, test_atr, test_classes) + erroArvoreCli
        train_atr, test_atr, train_classes, test_classes = train_test_split(baseBank.atributos, baseBank.classes, test_size=0.5, random_state=i) 
        qt2 = len(test_classes)
        erroKNNBank = classicarKNN(train_atr, train_classes, test_atr, test_classes) + erroKNNBank 
        erroNaiveBank  = naiveBayes(train_atr, train_classes, test_atr, test_classes) + erroNaiveBank 
        erroArvoreBank  = arvoreDecisao(train_atr, train_classes, test_atr, test_classes) + erroArvoreBank
    print("SEM PCA base Climate:\nerro KNN:%s\nacerto  NaiveBayes:%s\nerro arvore:%s\n"%(1-erroKNNCli/qt1,1-erroNaiveCli/qt1,1-erroArvoreCli/qt1))
    print("SEM PCA base BankNote:\nerro KNN:%s\nacerto  NaiveBayes:%s\nerro arvore:%s\n"%(1-erroKNNBank/qt2,1-erroNaiveBank/qt2,1-erroArvoreBank/qt2))
    
    acertoPCA = [[],[],[]]
    acertoPCAS = [[],[],[]]
    acertoPCASK = [[],[],[]]
    acertoPCACloves = [[],[],[]]
    acertoPCAScoreCloves = []
    extraido = list(range(1,19))
    pcaR = PCAR()
    for j in extraido:
        erroKNNCli = 0
        erroNaiveCli = 0
        erroArvoreCli = 0
        
        for i in range(100):
            
            train_atr, test_atr, train_classes, test_classes = train_test_split(baseClimate.atributos, baseClimate.classes, test_size=0.5, random_state=i) 
            pcaR.fitScore(Base(train_classes,train_atr))
            baseTreino = pcaR.run(Base(train_classes,train_atr), j)
            baseTeste = pcaR.run(Base(test_classes,test_atr),j)
            qt1 = len(test_classes)
            erroKNNCli = classicarKNN(baseTreino.atributos, baseTreino.classes, baseTeste.atributos, baseTeste.classes)/qt1 + erroKNNCli
        print("COM PCA base Climate - atr:%s:\nacerto KNN:%s\nacerto NaiveBayes:%s\nacerto arvore:%s\n"%(j,1-erroKNNCli,1-erroNaiveCli,1-erroArvoreCli))
        acertoPCA[0].append(1-erroKNNCli)
    
    print("\n") 

    GerarGrafico.gerarGrafico(extraido, acertoPCA[0], "Climate KNN PCA", "Extraido", "Acerto")
    #GerarGrafico.gerarGrafico(extraido,acertoPCAS[0],"Climate KNN PCA Score", "Extraido", "Acerto")
    #GerarGrafico.gerarGrafico(extraido,acertoPCASK[0],"Climate KNN PCA SK", "Extraido", "Acerto")
    GerarGrafico.gerarGrafico(extraido,acertoPCACloves[0],"Climate KNN PCA cloves", "Extraido", "Acerto")
    GerarGrafico.saveMultuplos(extraido, [acertoPCA[0]], "taxa de acerto PCAs", "quantidade de atributos", "acerto", ["PCA Raul"])
    extraido = list(range(1,19))
    
    pass