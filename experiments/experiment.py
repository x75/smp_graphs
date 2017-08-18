
import sys
from smp_graphs.experiment import get_args
from smp_graphs.experiment import Experiment, Graphviz
# from pyqtgraph.Qt import QtGui, QtCore

def main(args):

    experiment  = Experiment(args)

    experiment.run()

def main_graphviz(args):

    experiment = Graphviz(args)

    experiment.run()
    
if __name__ == "__main__":
    
    import signal
    def handler(signum, frame):
        print ('Signal handler called with signal', signum)
        # al.savelogs()
        # l.isrunning = False
        # if not args.batch:
        import rospy
        rospy.signal_shutdown("ending")
        sys.exit(0)
        # raise IOError("Couldn't open device!")

    signal.signal(signal.SIGINT, handler)

    
    args = get_args()
    modes = {
        'run': main,
        'graphviz': main_graphviz
        }
        
    assert args.mode in modes.keys()
    
    modes[args.mode](args)

    # Qt foo from pyqtgraph
    # def myExitHandler():
    #     print "exithandler"
    #     # return
    
    # app = QtGui.QApplication([]) # .instance().exec_()
    # app.aboutToQuit.connect(myExitHandler)
    # if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
    #     QtGui.QApplication.instance().exec_()
    # sys.exit()
    # app.exec_()
