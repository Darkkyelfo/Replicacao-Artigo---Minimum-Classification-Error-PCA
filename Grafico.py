'''
Created on 16 de out de 2017

@author: raul
'''
import matplotlib.pyplot as plt
class GerarGrafico(object):
    '''
    classdocs
    '''
    @staticmethod
    def gerarGrafico(x,y,titulo="",legendaX="",legendaY=""):
        plt.plot(x,y)
        plt.title(titulo)
        plt.xlabel(legendaX)
        plt.ylabel(legendaY)
        plt.grid(True)
        axes = plt.gca()
        axes.set_xlim([0,18])
        axes.set_ylim([0.76,0.87])
        plt.show()
        
    
    @staticmethod
    def saveMultuplos(x,Y,titulo="",legendaX="",legendaY="",legenda= []):
        for i in range(len(Y)):
            plt.plot(x,Y[i])
            plt.title(titulo)
            plt.xlabel(legendaX)
            plt.ylabel(legendaY)
            plt.grid(True)
        axes = plt.gca()
        axes.set_xlim([0,18])
        axes.set_ylim([0.76,0.87])
        plt.legend(legenda,loc='upper left')
        plt.show()
        