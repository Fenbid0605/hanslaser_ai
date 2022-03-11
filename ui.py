import logging
import sys

from PyQt6 import QtWidgets
from PyQt6.QtCore import QThread, pyqtSignal, QObject, pyqtSlot

from evolution_algorithm.main import GA, Predicted
from mainwindow import Ui_MainWindow
from torch import Tensor


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.progressBar.setValue(0)
        self.doButton.clicked.connect(self.__on_button_clicked)

        self.worker = PredictWorker()
        self.worker.predicted.connect(self.__on_predicted)
        self.worker.progress_changed.connect(self.__on_progress_changed)
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.start()

    @pyqtSlot()
    def __on_button_clicked(self):
        self.progressStatusLabel.setText('正在计算')
        self.outputLabel.clear()
        value = Tensor(
            [self.LdoubleSpinBox.value(), self.AdoubleSpinBox.value(), self.BdoubleSpinBox.value()])
        self.worker.predict.emit(value)

    @pyqtSlot(Predicted)
    def __on_predicted(self, predicted: Predicted):
        self.outputLabel.setText(f"电流: {predicted.current} 打标速度: {predicted.speed} Q频: {predicted.frequency} "
                                 f"Q释放: {predicted.release} 预测LAB: {predicted.L}, {predicted.A}, {predicted.B}")
        self.progressStatusLabel.setText('计算完成')

    def __on_progress_changed(self, i):
        self.progressBar.setValue(i)
        self.progressLabel.setText('%s%%' % i)


class PredictWorker(QObject):
    predict = pyqtSignal(object)
    predicted = pyqtSignal(Predicted)
    progress_changed: pyqtSignal

    def __init__(self, parent=None):
        super(PredictWorker, self).__init__(parent)
        self.ga = GA()
        self.progress_changed = self.ga.progress_changed
        self.predict.connect(self.__predict)

    @pyqtSlot(object)
    def __predict(self, value: Tensor):
        self.predicted.emit(self.ga.predict(value))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()

    sys.exit(app.exec())
