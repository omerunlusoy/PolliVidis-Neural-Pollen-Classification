# -*- coding: utf-8 -*-
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

class SayHi(QObject):
    
    
    def __init__(self):
        QObject.__init__(self)