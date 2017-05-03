
from smp_graphs.experiment import get_args
from smp_graphs.experiment import Experiment, Graphviz

def main(args):

    experiment  = Experiment(args)

    experiment.run()

def main_graphviz(args):

    experiment = Graphviz(args)

    experiment.run()
    
if __name__ == "__main__":
    args = get_args()
    modes = {
        'run': main,
        'graphviz': main_graphviz
        }
        
    assert args.mode in modes.keys()
    
    modes[args.mode](args)
