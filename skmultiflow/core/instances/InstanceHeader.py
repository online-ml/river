__author__ = 'Guilherme Matsumoto'

class InstanceHeader:
    '''
        Instance Header class
        -----------------------------
        Stores the header from an instance, simply keeps feature and label's names
    '''
    def __init__(self, header = None):
        self.header = header
        pass

    def getHeaderLabelAt(self, headerIndex = -1):
        return self.header[headerIndex] if (headerIndex > -1) else None
