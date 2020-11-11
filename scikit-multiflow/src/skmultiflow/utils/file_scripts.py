def clean_header(base_file, new_file=None, ignore_char='#'):
    """ clean_header
    
    Cleans a file header based on an ignore char passed as parameter.
    It removes the entire line that is followed by the ignore char.
    
    If the new_file parameter is not passed, the cleaned content is 
    going to be put in a file of the same name as the base file, but 
    concatenated with the string '_no_header'.
    
    Parameters
    ----------
    base_file: string
        The file to be cleaned.
    
    new_file: string, optional
        If given, this specifies the name of the new file, where 
        the changed content is going to be put.
    
    ignore_char: char
        The char that marks the beginning of a header/comment line.
         
    """
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
                if line[0] == ignore_char:
                    pass
                else:
                    out_file.write(line)


class RemoveHeader(object):
    """ RemoveHeader
        
    Class whose sole purpose is to remove the header from any file, and save 
    the changes to a new file.
    
    Parameters
    ----------
    base_file: string 
        The file to be cleaned.
        
    new_file: string, optional 
        If given, this specifies the name of the new file, where 
        the changed content is going to be put.
        
    ignore_char: char
        The char that marks the beginning of a header/comment line.
    
    Notes
    -----
    If the new_file parameter is not passed, the cleaned content is 
    going to be put in a file of the same name as the base file, but 
    concatenated with the string '_no_header'.
    
    """

    def __init__(self, base_file, new_file=None, ignore_char='#'):
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
        info = type(self).__name__ + ':'
        info += ' - base_file: {}'.format(self.base_file)
        info += ' - new_file: {}'.format(self.new_file)
        info += ' - ignore_char: {}'.format(self.ignore_char)
        return info

    def get_class_type(self):
        return 'file_utils'

    def clean_file(self):
        with open(self.base_file, 'r') as in_file:
            with open(self.new_file, 'w+') as out_file:
                for line in in_file:
                    if line[0] == self.ignore_char:
                        pass
                    else:
                        out_file.write(line)
