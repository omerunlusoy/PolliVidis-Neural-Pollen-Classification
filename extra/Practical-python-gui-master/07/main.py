import sys
from PyQt5.QtGui import QGuiApplication
from PyQt5.QtQml import QQmlApplicationEngine
from timer import Timer


def cleanUp():
    timer_object.app_running = False


app = QGuiApplication(sys.argv)
engine = QQmlApplicationEngine()
timer_object = Timer()
engine.rootContext().setContextProperty('timerObject', timer_object)
engine.load('main.qml')
engine.quit.connect(app.quit)
app.aboutToQuit.connect(cleanUp)
sys.exit(app.exec_())