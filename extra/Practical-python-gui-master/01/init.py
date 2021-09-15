# -*- coding: utf-8 -*-
import sys
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtQml import QQmlApplicationEngine
from hi import SayHi

app = QGuiApplication(sys.argv)
say_hi = SayHi()
engine = QQmlApplicationEngine()
engine.rootContext().setContextProperty('hiObject', say_hi)
engine.load('main.qml')
engine.quit.connect(app.quit)
sys.exit(app.exec_())