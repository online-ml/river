__author__ = 'Guilherme Matsumoto'

from skmultiflow.options.base_option import BaseOption

class NumericOption(BaseOption):
    def __init__(self, option_type ="NUM", option_name = None, option_cli_Parser = None, option_value = None):
        super().__init__()
        self.option_name = option_name
        self.option_cli_parser = option_cli_Parser
        self.option_value = option_value
        self.option_type = option_type
        pass

    def get_name(self):
        return self.option_name

    def get_value(self):
        return self.option_value

    def get_cli_char(self):
        return self.option_cli_parser

    def get_option_type(self):
        return self.option_type

    def set_value_via_cli_string(self, cli_string = None):
        pass

    def get_cli_option_from_dictionary(self):
        pass

    def get_info(self):
        pass