__author__ = 'Guilherme Matsumoto'

from skmultiflow.core.BaseObject import BaseObject

def clean_header(base_file, new_file=None, ignore_char='#'):
    # Find the new_file name
    if new_file is not None:
        nf = new_file
    else:
        path = base_file.split(".")
        base_path = path[0]
        if len(path) > 1:
            file_type = path[1]
        else:
            file_type = None
        base_path += '_no_header'
        nf = base_path + '.' + file_type if file_type is not None else base_path

    # Clean the file
    with open(base_file, 'r') as in_file:
        with open(nf, 'w+') as out_file:
            for line in in_file:
                if (line[0] == ignore_char):
                    pass
                else:
                    out_file.write(line)


class RemoveHeader(BaseObject):
    def __init__(self, base_file, new_file = None, ignore_char = '#'):
        super().__init__()
        # default values
        self.base_file = None
        self.new_file = None
        self.ignore_char = None

        self.configure(base_file, new_file, ignore_char)

    def configure(self, base_file, new_file, ignore_char):
        self.base_file = base_file
        self.ignore_char = ignore_char
        if new_file is not None:
            self.new_file = new_file
        else:
            path = self.base_file.split(".")
            base_path = path[0]
            if len(path) > 1:
                file_type = path[1]
            else:
                file_type = None
            base_path += '_no_header'
            self.new_file = base_path + '.' + file_type if file_type is not None else base_path


    def get_info(self):
        return 'Remove Header: base_file: ' + self.base_file + '  -  new_file: ' + self.new_file + \
               '  -  ignore_char: ' + self.ignore_char

    def get_class_type(self):
        return 'file_utils'

    def clean_file(self):
        with open(self.base_file, 'r') as in_file:
            with open(self.new_file, 'w+') as out_file:
                for line in in_file:
                    if (line[0] == self.ignore_char):
                        pass
                    else:
                        out_file.write(line)


if __name__ == '__main__':
    remover = RemoveHeader('~/Desktop/PRE/scikit-multiflow/n10.csv', None, '#')
    print('base: ' + remover.base_file)
    print('new: ' + remover.new_file)
    remover.clean_file()
    print('done')