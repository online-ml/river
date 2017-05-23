__author__ = 'Guilherme Matsumoto'

import skmultiflow.tasks.testMain as t
import sys, getopt

def demo(argv):
    t.demo_random_tree_gen(argv)
    #t.demo_file_stream()
    return None

if __name__ == '__main__':

    demo(sys.argv[1:])