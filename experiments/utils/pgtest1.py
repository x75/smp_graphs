import pyqtgraph as pg
import numpy as np
from pyqtgraph.Qt import QtGui, QtCore

x = np.random.normal(size=1000)
y = np.random.normal(size=1000)
pg.plot(x, y, pen=None, symbol='o')  ## setting pen=None disables line drawing


# pg.show()

## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    import sys
    # if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    #     QtGui.QApplication.instance().exec_()

    # def myExitHandler():
    #     print "myExitHandler"
    
    # app = QtGui.QApplication([]) # .instance().exec_()
    # app.aboutToQuit.connect(myExitHandler)
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
        # sys.exit(app.exec_())
