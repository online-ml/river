from skmultiflow.options.base_option import Option


class FileOption(Option):
    """ FileOption
    
    This class keeps information about a file.
    
    Parameters
    ----------
    option_type: string 
        The type of file related to the option. Can be anything.
        
    option_name: string 
        The option identification. Can be anything.
    
    option_value: string 
        The complete path to the file.
    
    file_extension: string 
        The type of extension.
    
    is_out: bool
        Whether it's an output file or not.
    
    """

    def __init__(self, option_type=None, option_name=None, option_value=None, file_extension=None, is_out=False):
        super().__init__()
        self.option_name = option_name
        self.option_value = option_value
        self.option_type = option_type
        self.file_type = file_extension
        self.is_output = is_out
        self.file_name = self.option_value

    def get_name(self):
        return self.option_name

    def get_file_name(self):
        return self.file_name

    def get_value(self):
        return self.option_value

    def get_option_type(self):
        return self.option_type

    def get_cli_char(self):
        return self.option_value

    def is_output(self):
        return self.is_output

    def get_info(self):
        return 'FileOption: option_type: ' + str(self.option_type) + \
               ' - option_name: ' + str(self.option_name) + \
               ' - option_value: ' + str(self.option_value) + \
               ' - file_extension: ' + str(self.file_type) + \
               ' - is_out: ' + ('True' if self.is_output else 'False')
