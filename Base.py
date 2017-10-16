'''
Created on Aug 14, 2017

@author: raul
'''

class Base(object):
    '''
    classdocs
    '''

    
    def __init__(self, classes,atributos):
        self.tiposClasses = []
        self.classes = classes
        self.atributos = atributos
        for e in classes:
            if(e not in self.tiposClasses):
                self.tiposClasses.append(e) 
        