__author__ = 'Guilherme Matsumoto'

from skmultiflow.core.instances.InstanceHeader import InstanceHeader
from skmultiflow.core.instances.InstanceData import InstanceData
from skmultiflow.data.BaseInstanceStream import BaseInstanceStream
from skmultiflow.core.instances.Instance import Instance
import numpy as np
from array import array
'''
    keep track of the last instance generated
'''


class RandomTreeGenerator(BaseInstanceStream):
    '''
        RandomTreeGenerator
        ---------------------------------------------
        Instance generator based on a random tree that splits attributes and sets labels to its leafs.
        Supports the addition of noise.
        
        Parser parameters
        ---------------------------------------------
        -r: Seed for random generation of tree (Default: 23)
        -i: Seed for random generation of instances (Default: 12)
        -c: The number of classes to generate (Default: 2)
        -o: The number of nominal attributes to generate (Default: 5)
        -u: The number of numerical attributes to generate (Default: 5)
        -v: The number of values to generate per nominal attribute (Default: 5)
        -d: The maximum depth of the tree concept (Default: 5)
        -l: The first level of the tree above MaxTreeDepth that can have leaves (Default: 3)
        -f: The fraction of leaves per level from FirstLeafLevel onwards (Default: 0.15)
        
        Instance Data Format
        --------------------------------------------
        The data coming from this generator follows a strict form:
         ___________________________________________________________
        |   Numeric Attributes  |   Nominal Attributes  |   Class   |
         -----------------------------------------------------------
    '''
    def __init__(self, optList = None):
        super().__init__()

        # default values
        self.randomTreeSeed = 23
        self.randomInstanceSeed = 12
        self.numClasses = 2
        self.numNumericalAttributes = 5
        self.numNominalAttributes = 5
        self.numValuesPerNominalAtt = 5
        self.maxTreeDepth = 5
        self.minLeafDepth = 3
        self.fractionOfLeavesPerLevel = 0.15
        self.instanceIndex = 0
        self.treeRoot = None
        self.instanceRandom = None
        self.classHeader = None
        self.attributesHeader = None
        self.instanceRandom = None
        self.configure(optList)
        pass

    def configure(self, optList):
        if optList is not None:
            for i in range(len(optList)):
                opt, arg = optList[i]
                if opt in ("-r"):
                    self.randomTreeSeed = int(arg)
                elif opt in ("-i"):
                    self.randomInstanceSeed = int(arg)
                elif opt in ("-c"):
                    self.numClasses = int(arg)
                elif opt in ("-o"):
                    self.numNominalAttributes = int(arg)
                elif opt in ("-u"):
                    self.numNumericalAttributes = int(arg)
                elif opt in ("-v"):
                    self.numValuesPerNominalAtt = int(arg)
                elif opt in ("-d"):
                    self.maxTreeDepth = int(arg)
                elif opt in ("-l"):
                    self.minLeafDepth = int(arg)
                elif opt in ("-f"):
                    self.fractionOfLeavesPerLevel = float(arg)
        self.classHeader = InstanceHeader(["class"])
        self.attributesHeader = []
        for i in range(self.numNumericalAttributes):
            self.attributesHeader.append("NumAtt" + str(i))
        for i in range(self.numNominalAttributes):
            for j in range(self.numValuesPerNominalAtt):
                self.attributesHeader.append("NomAtt" + str(i) + "_Val" + str(j))

        print(self.attributesHeader)

        pass

    def prepareForUse(self):
        self.instanceRandom = np.random
        self.instanceRandom.seed(self.randomInstanceSeed)
        self.restart()

    def generateRandomTree(self):
        treeRand = np.random
        treeRand.seed(self.randomTreeSeed)
        nominalAttCandidates = array('i')

        minNumericVals = array('d')
        maxNumericVals = array('d')
        for i in range(self.numNumericalAttributes):
            minNumericVals.append(0.0)
            maxNumericVals.append(1.0)

        for i in range(self.numNominalAttributes + self.numNominalAttributes):
            nominalAttCandidates.append(i)

        self.treeRoot = self.generateRandomTreeNode(0, nominalAttCandidates, minNumericVals, maxNumericVals, treeRand)


    def generateRandomTreeNode(self, currentDepth, nominalAttCandidates, minNumericVals, maxNumericVals, rand):
        if ((currentDepth >= self.maxTreeDepth) | ((currentDepth >= self.minLeafDepth) & (self.fractionOfLeavesPerLevel >= (1.0 - rand.rand())))):
            leaf = Node()
            leaf.classLabel = rand.randint(0, self.numClasses)
            return leaf

        node = Node()
        chosenAtt = rand.randint(0, len(nominalAttCandidates))
        if (chosenAtt < self.numNumericalAttributes):
            numericIndex = chosenAtt
            node.splitAttIndex = numericIndex
            minVal = minNumericVals[numericIndex]
            maxVal = maxNumericVals[numericIndex]
            node.splitAttValue = ((maxVal - minVal) * rand.rand() + minVal)
            node.children = []

            newMaxVals = maxNumericVals[:]
            newMaxVals[numericIndex] = node.splitAttValue
            node.children.append(self.generateRandomTreeNode(currentDepth+1, nominalAttCandidates, minNumericVals, newMaxVals, rand))

            newMinVals = minNumericVals[:]
            newMinVals[numericIndex] = node.splitAttValue
            node.children.append(self.generateRandomTreeNode(currentDepth+1, nominalAttCandidates, newMinVals, maxNumericVals, rand))
        else:
            node.splitAttIndex = nominalAttCandidates[chosenAtt]
            newNominalCandidates = array('d', nominalAttCandidates)
            newNominalCandidates.remove(node.splitAttIndex)

            for i in range(self.numValuesPerNominalAtt):
                node.children.append(self.generateRandomTreeNode(currentDepth+1, newNominalCandidates, minNumericVals, maxNumericVals, rand))

        return node

    def classifyInstance(self, node, attVals):
        if len(node.children) == 0:
            return node.classLabel
        if node.splitAttIndex < self.numNumericalAttributes:
            aux = 0 if attVals[node.splitAttIndex] < node.splitAttValue else 1
            return self.classifyInstance(node.children[aux], attVals)
        else:
            return self.classifyInstance(node.children[self.getIntegerNominalAttributeRepresentation(node.splitAttIndex, attVals)], attVals)

    def getIntegerNominalAttributeRepresentation(self, nominalIndex = None, attVals = None):
        '''
            The nominalIndex uses as reference the number of nominal attributes plus the number of nominal attributes.
             In this way, to find which 'hot one' variable from a nominal attribute is active, we do some basic math.
             This function returns the index of the active variable in a nominal attribute 'hot one' representation.
        '''
        minIndex = self.numNumericalAttributes + (nominalIndex - self.numNumericalAttributes)*self.numValuesPerNominalAtt
        for i in range(self.numValuesPerNominalAtt):
            if attVals[int(minIndex)] == 1:
                return i
            minIndex += 1
        return None

    def estimatedRemainingInstances(self):
        return -1

    def hasMoreInstances(self):
        return True

    def nextInstance(self):
        att = array('d')
        for i in range(self.numNumericalAttributes):
            att.append(self.instanceRandom.rand())

        for i in range(self.numNominalAttributes):
            aux = self.instanceRandom.randint(0, self.numValuesPerNominalAtt)
            for j in range(self.numValuesPerNominalAtt):
                if aux == j:
                    att.append(1.0)
                else:
                    att.append(0.0)

        att.append(self.classifyInstance(self.treeRoot, att))
        inst = Instance(self.numNominalAttributes*self.numValuesPerNominalAtt + self.numNumericalAttributes, self.numClasses, -1, att)
        return inst

    def isRestartable(self):
        return True

    def restart(self):
        self.instanceIndex = 0
        self.instanceRandom = np.random
        self.instanceRandom.seed(self.randomInstanceSeed)
        self.generateRandomTree()
        pass

    def nextInstanceMiniBatch(self):
        pass

    def hasMoreMiniBatch(self):
        pass

    def getNumNominalAttributes(self):
        return self.numNominalAttributes

    def getNumNumericalAttributes(self):
        return self.numNumericalAttributes

    def getNumValuesPerNominalAttribute(self):
        return self.numValuesPerNominalAtt

    def getNumClasses(self):
        return self.numClasses

    def getAttributesHeader(self):
        return self.attributesHeader

    def getClassesHeader(self):
        return self.classHeader

class Node:
    def __init__(self, classLabel = None, splitAttIndex = None, splitAttValue = None):
        self.classLabel = classLabel
        self.splitAttIndex = splitAttIndex
        self.splitAttValue = splitAttValue
        self.children = []