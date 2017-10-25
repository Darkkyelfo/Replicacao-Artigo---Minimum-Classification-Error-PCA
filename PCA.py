'''
Created on Aug 29, 2017

@author: raul
'''
import numpy as np
from Base import Base
from funcoesAuxiliares import separarElementosPorClasse
import math
import copy

class PCA(object):
    
    def run (self,base1,k="T"):
        if(k=="T"):
            k = len(base1.atributos[0])

        #base = Base(copy.deepcopy(base1.classes),copy.deepcopy(base1.atributos))
        #media = np.mean(np.array(base1.atributos),axis=0)
        copia = np.array(copy.deepcopy(base1.atributos))
        #cria a matriz de subtração
        #subtracao = (copia - media)
        #cov = np.cov(copia.T)
        #autoValues,autoVectors = np.linalg.eig(cov)
        #autoVectors = autoVectors.T
        #autoValues,autoVectors = zip(*sorted(zip(autoValues, autoVectors.T ),reverse=True))

        #temp = []
        #for i in range(len(autoValues)):
            #temp.append([autoValues[i],autoVectors[i]])
        #temp.sort(key=itemgetter(0),reverse=True)
        #autoVectors = []
        #for i in temp:
            #autoVectors.append(i[1])

        autoVectors = self.autoVectors[0:k]
        #autoValues = autoValues[0:len(base.atributos[0])-k]
        novosAtributos = np.dot(copia,np.array(autoVectors).T)
        return (Base(base1.classes,novosAtributos))
    
    def fit(self,atrTreino):
        copia = np.array(copy.deepcopy(atrTreino))
        cov = np.cov(copia.T)
        autoValues,autoVectors = np.linalg.eig(cov)
        autoVectors = autoVectors.T
        self.autoValues,self.autoVectors = zip(*sorted(zip(autoValues, autoVectors),reverse=True))
        return autoValues,autoVectors
        
    '''
    @jit
    def projetar(self,autovetores,atributos):
        novosAtributos = []
        for pca in autovetores:
            temp = []
            for atr in atributos:
                temp.append(np.vdot(atr,pca))
            novosAtributos.append(temp)
        return np.array(novosAtributos).T
    '''
    def pcaScore(self,base1,k="T"):
        if(k=="T"):
            k = len(base1.atributos[0])
        base = Base(copy.deepcopy(base1.classes),copy.deepcopy(base1.atributos))
        media = np.mean(np.array(base.atributos),axis=0)
        copia = np.array(copy.deepcopy(base.atributos))
        subtracao = (copia - media)
        cov = np.cov(copia.T)
        autoValues,autoVectors = np.linalg.eig(cov)
        m1,m2 = separarElementosPorClasse(base,base.classes)
        m1 = np.mean(m1, axis=0)
        m2 = np.mean(m2, axis=0)
        scores = self.__score(m1, m2, autoValues)
        scores,autoVectors = zip(*sorted(zip(scores, autoVectors ),reverse=True))
        autoVectors = autoVectors[0:k]
        novosAtributos = np.dot(subtracao,np.array(autoVectors).T)
        return (Base(base.classes,novosAtributos))
    
    
    def __score(self,media1,media2,autovalores):
        scores = []
        for i in range(len(media1)):
            s = 0
            if(autovalores[i] != 0):
                s = math.fabs((media1[i]-media2[i]))/autovalores[i]
            scores.append(s)
        return scores

  
               
    
    