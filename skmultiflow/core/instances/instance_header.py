__author__ = 'Guilherme Matsumoto'

from skmultiflow.core.base_object import BaseObject


class InstanceHeader(BaseObject):
    '''
        Instance Header class
        -----------------------------
        Stores the header from an instance, simply keeps feature and label's names
    '''
    def __init__(self, header = None):
        super().__init__()
        self.header = header
        pass

    def get_header_label_at(self, headerIndex = -1):
        return self.header[headerIndex] if (headerIndex > -1) else None

    def get_class_type(self):
        return 'instance'

    def get_info(self):
        pass
