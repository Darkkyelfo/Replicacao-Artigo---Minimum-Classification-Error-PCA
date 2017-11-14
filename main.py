'''
Created on 15 de out de 2017

@author: raul
'''
from CreateBaseFromFile import CreateBaseFromFile
from sklearn.model_selection import train_test_split
from PCA import PCA as PCAR
from PCA import PCA_SCORE as PCARS 
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
    
    acertoPCA = [[],[],[],[]]
    acertoPCAS = [[],[],[],[]]
    extraido = list(range(1,19))
    for j in extraido:
        erros = [0]*4
        errosScore = [0]*4
        for i in range(100):
            pcaR = PCAR()
            pcaRS = PCARS()
            train_atr, test_atr, train_classes, test_classes = train_test_split(baseClimate.atributos, baseClimate.classes, test_size=0.5, random_state=i) 
            b = Base(train_classes,train_atr,{0:0,1:1})#cria a base de treino
            pcaR.fit(b)#prepara o PCA
            pcaRS.fit(b)#prepara o PCA com score
            
            baseTreino = pcaR.run(Base(train_classes,train_atr,{0:0,1:1}), j) #Cria base de treino projetada pelo PCA
            baseTeste = pcaR.run(Base(test_classes,test_atr,{0:0,1:1}),j) #Cria base de teste projetada pelo PCA
            
            baseTreinoS = pcaRS.run(Base(train_classes,train_atr,{0:0,1:1}), j) #Cria base de treino projetada pelo PCA com score
            baseTesteS = pcaRS.run(Base(test_classes,test_atr,{0:0,1:1}),j) #Cria base de teste projetada pelo PCA com score
            qt1 = len(test_classes)
            #Erros dos classificados - PCA
            erros[0] = classicarKNN(baseTreino.atributos, baseTreino.classes, baseTeste.atributos, baseTeste.classes)/qt1 + erros[0]
            erros[1] = naiveBayes(baseTreino.atributos, baseTreino.classes, baseTeste.atributos, baseTeste.classes)/qt1 + erros[1]
            erros[2] = arvoreDecisao(baseTreino.atributos, baseTreino.classes, baseTeste.atributos, baseTeste.classes)/qt1 + erros[2]
            erros[3] = dlFisher(baseTreino.atributos, baseTreino.classes, baseTeste.atributos, baseTeste.classes)/qt1 + erros[3]
            #Erros dos classificadores - PCA Score
            errosScore[0] = classicarKNN(baseTreinoS.atributos, baseTreinoS.classes, baseTesteS.atributos, baseTesteS.classes)/qt1 + errosScore[0]
            errosScore[1] = naiveBayes(baseTreinoS.atributos, baseTreinoS.classes, baseTesteS.atributos, baseTesteS.classes)/qt1 + errosScore[1]
            errosScore[2] = arvoreDecisao(baseTreinoS.atributos, baseTreinoS.classes, baseTesteS.atributos, baseTesteS.classes)/qt1 + errosScore[2]
            errosScore[3] = dlFisher(baseTreinoS.atributos, baseTreinoS.classes, baseTesteS.atributos, baseTesteS.classes)/qt1 + errosScore[3]
            
        print("COM PCA base Climate - atr:%s:\nacerto KNN:%s\nacerto NaiveBayes:%s\nacerto arvore:%s\nfisher:%s\n"%(j,1-erros[0],1-erros[2],1-erros[2],1-erros[3]))
        print("COM PCA Score base Climate - atr:%s:\nacerto KNN:%s\nacerto NaiveBayes:%s\nacerto arvore:%s\nfisher:%s\n"%(j,1-errosScore[0],1-errosScore[1],1-errosScore[2],1-errosScore[3]))
        for i,e in enumerate(erros):
            acertoPCA[i].append(1-e)
            acertoPCAS[i].append(1-errosScore[i])
    
    print("\n") 
    GerarGrafico.saveMultuplos(extraido, [acertoPCA[0],acertoPCAS[0]], "taxa de acerto PCAs - KNN CLIMATE", "quantidade de atributos", "acerto", ["PCA","PCA Score"])
    GerarGrafico.saveMultuplos(extraido, [acertoPCA[1],acertoPCAS[1]], "taxa de acerto PCAs - Naive Bayes CLIMATE", "quantidade de atributos", "acerto", ["PCA","PCA Score"])
    GerarGrafico.saveMultuplos(extraido, [acertoPCA[2],acertoPCAS[2]], "taxa de acerto PCAs - Arvore CLIMATE", "quantidade de atributos", "acerto", ["PCA","PCA Score"])
    GerarGrafico.saveMultuplos(extraido, [acertoPCA[3],acertoPCAS[3]], "taxa de acerto PCAs - Fisher CLIMATE", "quantidade de atributos", "acerto", ["PCA","PCA Score"])
    
    extraido = list(range(1,5))
    acertoPCA = [[],[],[],[]]
    acertoPCAS = [[],[],[],[]]
    for j in extraido:
            erros = [0]*4
            errosScore = [0]*4
            for i in range(100):
                pcaR = PCAR()
                pcaRS = PCARS()
                train_atr, test_atr, train_classes, test_classes = train_test_split(baseBank.atributos, baseBank.classes, test_size=0.5, random_state=i) 
                b = Base(train_classes,train_atr)
                pcaR.fit(b)
                pcaRS.fit(b)
                baseTreino = pcaR.run(Base(train_classes,train_atr), j)
                baseTeste = pcaR.run(Base(test_classes,test_atr),j)
                
                baseTreinoS = pcaRS.run(Base(train_classes,train_atr), j)
                baseTesteS = pcaRS.run(Base(test_classes,test_atr),j)
                qt1 = len(test_classes)
                
                erros[0] = classicarKNN(baseTreino.atributos, baseTreino.classes, baseTeste.atributos, baseTeste.classes)/qt1 + erros[0]
                erros[1] = naiveBayes(baseTreino.atributos, baseTreino.classes, baseTeste.atributos, baseTeste.classes)/qt1 + erros[1]
                erros[2] = arvoreDecisao(baseTreino.atributos, baseTreino.classes, baseTeste.atributos, baseTeste.classes)/qt1 + erros[2]
                erros[3] = dlFisher(baseTreino.atributos, baseTreino.classes, baseTeste.atributos, baseTeste.classes)/qt1 + erros[3]
                
                errosScore[0] = classicarKNN(baseTreinoS.atributos, baseTreinoS.classes, baseTesteS.atributos, baseTesteS.classes)/qt1 + errosScore[0]
                errosScore[1] = naiveBayes(baseTreinoS.atributos, baseTreinoS.classes, baseTesteS.atributos, baseTesteS.classes)/qt1 + errosScore[1]
                errosScore[2] = arvoreDecisao(baseTreinoS.atributos, baseTreinoS.classes, baseTesteS.atributos, baseTesteS.classes)/qt1 + errosScore[2]
                errosScore[3] = dlFisher(baseTreinoS.atributos, baseTreinoS.classes, baseTesteS.atributos, baseTesteS.classes)/qt1 + errosScore[3]
                
            print("COM PCA base BANK - atr:%s:\nacerto KNN:%s\nacerto NaiveBayes:%s\nacerto arvore:%s\nfisher:%s\n"%(j,1-erros[0],1-erros[2],1-erros[2],1-erros[3]))
            print("COM PCA Score base BANK - atr:%s:\nacerto KNN:%s\nacerto NaiveBayes:%s\nacerto arvore:%s\nfisher:%s\n"%(j,1-errosScore[0],1-errosScore[1],1-errosScore[2],1-errosScore[3]))
            for i,e in enumerate(erros):
                acertoPCA[i].append(1-e)
                acertoPCAS[i].append(1-errosScore[i])
        
    print("\n") 
    GerarGrafico.saveMultuplos(extraido, [acertoPCA[0],acertoPCAS[0]], "taxa de acerto PCAs - KNN BANK", "quantidade de atributos", "acerto", ["PCA","PCA Score"],[0,4],[0.9,1.2])
    GerarGrafico.saveMultuplos(extraido, [acertoPCA[1],acertoPCAS[1]], "taxa de acerto PCAs - Naive Bayes BANK", "quantidade de atributos", "acerto", ["PCA","PCA Score"],[0,4],[0.9,1.2])
    GerarGrafico.saveMultuplos(extraido, [acertoPCA[2],acertoPCAS[2]], "taxa de acerto PCAs - Arvore BANK", "quantidade de atributos", "acerto", ["PCA","PCA Score"],[0,4],[0.9,1.2])
    GerarGrafico.saveMultuplos(extraido, [acertoPCA[3],acertoPCAS[3]], "taxa de acerto PCAs - Fisher BANK", "quantidade de atributos", "acerto", ["PCA","PCA Score"],[0,4],[0.9,1.2])
    pass