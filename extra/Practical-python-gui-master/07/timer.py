import threading
from time import strftime, gmtime, sleep
from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal

class Timer(QObject):


    def __init__(self):
        QObject.__init__(self)
        self.app_running = True


    loadUp = pyqtSignal(str, arguments=["setTime"])


    @pyqtSlot()
    def bootUp(self):


        print('ok')
        time_thread = threading.Thread(target=self.setTime)
        time_thread.daemon = True
        time_thread.start()


    @pyqtSlot()
    def setTime(self):
        print('second ok, ok')
        while self.app_running:
            print('we are running ok')
            timer = strftime('%H:%M:%S{}%A, %d %B %Y', gmtime()).format('<br/>')
            self.loadUp.emit(timer)
            sleep(0.25)
