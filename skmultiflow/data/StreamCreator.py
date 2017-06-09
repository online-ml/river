from skmultiflow.data.FileStream import FileStream
from skmultiflow.data.generators.WaveformGenerator import WaveformGenerator
from skmultiflow.data.generators.RandomTreeGenerator import RandomTreeGenerator

def CreateStreamFromArgumentDict(argument_list):
    if argument_list[0] == 'CsvFileStream':
        if len(argument_list) > 1:
            return FileStream(argument_list[1:])
        else:
            return FileStream()
    elif argument_list[0] == 'WaveformGenerator':
        if len(argument_list) > 1:
            return WaveformGenerator(argument_list[1:])
        else:
            return WaveformGenerator()
    elif argument_list[0] == 'RandomTreeGenerator':
        if len(argument_list) > 1:
            return RandomTreeGenerator(argument_list[1:])
        else:
            return RandomTreeGenerator()
    return None