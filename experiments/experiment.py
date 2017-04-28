
from smp_graphs.experiment import get_args
from smp_graphs.experiment import Experiment

def main(args):

    experiment  = Experiment(args)

    experiment.run()

if __name__ == "__main__":
    main(get_args())
