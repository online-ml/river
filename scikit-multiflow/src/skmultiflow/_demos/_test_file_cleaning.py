from skmultiflow._demos import _test_prequential
from skmultiflow.utils.file_scripts import clean_header
import logging


def demo(output_file='testlog.csv'):
    """ _test_file_cleaning
    
    This demo will run a the _test_prequential demo, which will generate a csv 
    output file named as the parameter output_file. This generated file will 
    then have its header removed.
    
    output_file: string
        The name of the output file.
     
    """
    _test_prequential.demo(output_file, 40000)
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    logging.info('Finished the Prequential evaluation...')
    logging.info('...')
    logging.info('Starting file cleaning...')
    clean_header(output_file)
    logging.info('File successfully cleaned')

if __name__ == '__main__':
    demo('test_file_cleaning.csv')