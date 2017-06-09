import sys, argparse

class ScikitStreamParser():
    def __init__(self):
        self.progName = ""
        pass

    def parse(self, argv):
        self.progName = argv[1]
        argv = argv[1:]
        opts = []
        args = []
        end = 0
        i=0
        while i in range(len(argv)):
            opts.append(argv[i])
            i+=1
            if argv[i][:1] == '(':
                end = self.getFinalNestIndex(argv, i)
                pass
            else:
                args.append(argv[(2*i)+1])
            i+=1
        return (opts, args)


    def getFinalNestIndex(self, argv, starting_index):
        i=0
        while i in range(len(argv[starting_index:])):
            if argv[i][-1] == ')':
                print("lol")
        pass

def demo(argv):
    parser = argparse.ArgumentParser(description='Testing argparse module')

    parser.add_argument("-l", dest='classifier', type=str, help='Classifier to train', default='NaiveBayes')
    parser.add_argument("-s", dest='stream', type=str, help='Stream to train', default='RandomTree')
    parser.add_argument("-e", dest='performance', type=str, help='Classification performance evaluation method')
    parser.add_argument("-i", dest='maxInt', type=int, help='Maximum number of instances')
    parser.add_argument("-t", dest='max_time', type=int, help='Max number of seconds')
    parser.add_argument("-f", dest='n_wait', type=int, help='How many instances between samples of the learning performance')
    parser.add_argument("-b", dest='maxSize', type=int, help='Maximum size of model')
    parser.add_argument("-O", dest='out', type=str, help='Output file')


    args = parser.parse_args()

    print(args)

    if (args.classifier is not None):
        split = args.classifier.split()
        print(split)

    if args.stream is not None:
        split = args.stream.split()
        if len(split) > 1:
            print(split)


    pass

def demo1(argv):
    print(argv)


if __name__ == '__main__':
    demo(sys.argv)

    #parser = ScikitStreamParser()
    #opts, args = parser.parse(sys.argv)
    #print(opts)
    #print(args)
