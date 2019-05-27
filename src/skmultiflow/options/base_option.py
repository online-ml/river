from abc import ABCMeta, abstractmethod


class Option(object, metaclass=ABCMeta):
    """ BaseOption
    
    The abstract class that defines the constraints for all option classes in this framework.
    
    Raises
    ------
    NotImplementedError: This is an abstract class.
    
    """

    def __init__(self):
        super().__init__()
        pass

    @abstractmethod
    def get_name(self):
        """ get_name
        
        Gets the option's name.
        
        Returns
        -------
        string 
            The option's name.
        
        """
        raise NotImplementedError

    @abstractmethod
    def get_value(self):
        """ get_value
        
        Gets the option's value.
        
        Returns
        -------
        string or numerical
            The option's value.
        
        """
        raise NotImplementedError

    @abstractmethod
    def get_cli_char(self):
        """ get_cli_char
        
        Get the parser string for the option.
        
        Returns
        -------
        char or string
            The parser string for the option.
        
        Notes
        -----
        Deprecated. Do not use this function.
        
        """
        raise NotImplementedError

    @abstractmethod
    def get_option_type(self):
        """ get_option_type
        
        Get the option's type.
        
        Returns
        -------
        string 
            The option's type.
        
        """
        raise NotImplementedError

    @abstractmethod
    def get_info(self):
        raise NotImplementedError
