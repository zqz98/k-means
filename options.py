import argparse

parser = argparse.ArgumentParser()
# Required parameters
parser.add_argument("--k", default=3, type=int,
                    help="The number of Cluster for k-means.")
parser.add_argument("--num", default=100, type=int,
                    help="The number of data.")
parser.add_argument("--dim", default=3, type=int,
                    help="The dim of data.")
parser.add_argument("--max", default=100, type=int,
                    help="Max value of data.")
parser.add_argument("--min", default=0, type=int,
                    help="min value of data.")
parser.add_argument("--iteration", default=1000, type=int,
                    help="Maximum number of iterations of the program .")
parser.add_argument("--error", default=0.001, type=float,
                    help="Program iteration termination error.")
parser.add_argument("--distance", default='calcDis_cos', type=str,choices=['calcDis_e', 'calcDis_cos'],
                    help="Distance calculation method")
parser.add_argument("--seed", default=1, type=int,
                    help="random seed for initialization")
args = parser.parse_args()