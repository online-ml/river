__author__ = 'Guilherme Matsumoto'

from skmultiflow.options.base_option import BaseOption

class FileOption(BaseOption):
    '''
        fileOption class
        --------------------------------------
        Maintain options concerning file management.
    '''
    def __init__(self, option_type = None, option_name = None, option_value = None, file_extension = None, is_out = False):
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

    def set_value_via_cli_string(self, cli_string = None):
        self.option_value = cli_string

    def get_cli_option_from_dictionary(self):
        return {"file_name" : "-n",
                "file_type" : "-t"}[self.option_type]

    def get_info(self):
        pass