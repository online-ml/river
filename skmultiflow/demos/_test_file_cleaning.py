__author__ = 'Guilherme Matsumoto'

from skmultiflow.demos import _test_prequential
from skmultiflow.core.utils.file_scripts import clean_header
import logging


def demo(output_file='testlog.csv'):
    _test_prequential.demo(output_file)
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    logging.info('Finished the Prequential evaluation...')
    logging.info('...')
    logging.info('Starting file cleaning...')
    clean_header(output_file)
    logging.info('File successfully cleaned')
    pass

if __name__ == '__main__':
    demo('log2.csv')