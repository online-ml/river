from skmultiflow.data.CsvFileStream import CsvFileStream
from skmultiflow.data.generators.WaveformGenerator import WaveformGenerator
from skmultiflow.data.generators.RandomTreeGenerator import RandomTreeGenerator

def CreateStreamFromArgumentDict(argumentList):
    if argumentList[0] == 'CsvFileStream':
        if len(argumentList) > 1:
            return CsvFileStream(argumentList[1:])
        else:
            return CsvFileStream()
    elif argumentList[0] == 'WaveformGenerator':
        if len(argumentList) > 1:
            return WaveformGenerator(argumentList[1:])
        else:
            return WaveformGenerator()
    elif argumentList[0] == 'RandomTreeGenerator':
        if len(argumentList) > 1:
            return RandomTreeGenerator(argumentList[1:])
        else:
            return RandomTreeGenerator()
    return None