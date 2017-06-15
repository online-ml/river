__author__ = 'Guilherme Matsumoto'

from skmultiflow.core.BaseObject import BaseObject


class FastBuffer(BaseObject):
    def __init__(self, max_size, object_list=None):
        super().__init__()
        #default values
        self.current_size = 0
        self.max_size = None
        self.buffer = []

        self.configure(max_size, object_list)

    def get_class_type(self):
        return 'data_structure'

    def configure(self, max_size, object_list):
        self.max_size = max_size
        if object_list is not None:
            self.buffer = object_list

    def add_element(self, element_list):
        if (self.current_size+len(element_list)) <= self.max_size:
            for i in range(len(element_list)):
                self.buffer.append(element_list[i])
            self.current_size += len(element_list)
            return None
        else:
            aux = []
            for i in range(len(element_list)):
                aux.append(self.buffer.pop(0))
                self.buffer.append(element_list[i])
            return aux

    def get_next_element(self):
        return self.buffer.pop(0)

    def clear_queue(self):
        self._clear_all()

    def _clear_all(self):
        del self.buffer[:]

    def print_queue(self):
        print(self.buffer)

    def isfull(self):
        return self.current_size == self.max_size

    def isempty(self):
        return self.current_size == 0

    def peek(self):
        try:
            return self.buffer[0]
        except IndexError:
            return None

    def get_queue(self):
        return self.buffer

if __name__ == '__main__':
    buffer = FastBuffer(10, None)
    for i in range(10):
        buffer.add_element(i)
    buffer.print_queue()
    buffer.add_element(15)
    buffer.add_element(99)
    buffer.print_queue()
    print(str(buffer.get_next_element()))
    buffer.print_queue()
