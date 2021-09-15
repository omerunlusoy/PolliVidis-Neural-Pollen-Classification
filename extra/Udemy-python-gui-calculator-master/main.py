import sys
from PyQt5.QtGui import QGuiApplication, QIcon
from PyQt5.QtQml import QQmlApplicationEngine
from func import Calculator

app = QGuiApplication(sys.argv)
app.setWindowIcon(QIcon('resources/images/disc_logo.png'))

calc = Calculator()

engine = QQmlApplicationEngine()
engine.rootContext().setContextProperty('Calculator', calc)
engine.load('resources/qml/main.qml')
engine.quit.connect(app.quit)

sys.exit(app.exec_())
