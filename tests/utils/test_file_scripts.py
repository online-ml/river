import os
import filecmp
import difflib
from shutil import copyfile
from skmultiflow.utils.file_scripts import RemoveHeader, clean_header


def test_clean_header(tmpdir, test_path):
    file_with_header = os.path.join(test_path, 'file_with_header.csv')
    file_without_header = os.path.join(test_path, 'file_without_header.csv')

    clean_file = os.path.join(str(tmpdir), "clean_file.csv")
    clean_header(base_file=file_with_header, new_file=clean_file)

    compare_files(clean_file, file_without_header)

    # Test remover without defining new file
    file_with_header_tmp = os.path.join(str(tmpdir), 'file_with_header.csv')
    file_without_header_tmp = os.path.join(str(tmpdir), 'file_with_header_no_header.csv')
    copyfile(file_with_header, file_with_header_tmp)

    clean_header(base_file=file_with_header_tmp)
    compare_files(clean_file, file_without_header_tmp)

    # Test remover without defining new file and without format
    file_with_header_tmp = os.path.join(str(tmpdir), 'file_with_header')
    file_without_header_tmp = os.path.join(str(tmpdir), 'file_with_header_no_header')
    copyfile(file_with_header, file_with_header_tmp)

    clean_header(base_file=file_with_header_tmp)
    compare_files(clean_file, file_without_header_tmp)


def test_remove_header(tmpdir, test_path):
    file_with_header = os.path.join(test_path, 'file_with_header.csv')
    file_without_header = os.path.join(test_path, 'file_without_header.csv')

    clean_file = os.path.join(str(tmpdir), "clean_file_class.csv")
    remover = RemoveHeader(base_file=file_with_header, new_file=clean_file)

    remover.clean_file()
    compare_files(clean_file, file_without_header)

    expected_info = 'RemoveHeader: - base_file: {} - new_file: {} - ignore_char: #'.format(file_with_header,
                                                                                           clean_file)
    assert remover.get_info() == expected_info

    assert remover.get_class_type() == 'file_utils'

    # Test remover without defining new file
    file_with_header_tmp = os.path.join(str(tmpdir), 'file_with_header.csv')
    file_without_header_tmp = os.path.join(str(tmpdir), 'file_with_header_no_header.csv')
    copyfile(file_with_header, file_with_header_tmp)

    remover = RemoveHeader(base_file=file_with_header_tmp)

    remover.clean_file()
    compare_files(clean_file, file_without_header_tmp)

    # Test remover without defining new file and without format
    file_with_header_tmp = os.path.join(str(tmpdir), 'file_with_header')
    file_without_header_tmp = os.path.join(str(tmpdir), 'file_with_header_no_header')
    copyfile(file_with_header, file_with_header_tmp)

    remover = RemoveHeader(base_file=file_with_header_tmp)

    remover.clean_file()
    compare_files(clean_file, file_without_header_tmp)


def compare_files(test, expected):
    lines_expected = open(expected).readlines()
    lines_test = open(test).readlines()

    print(''.join(difflib.ndiff(lines_test, lines_expected)))
    filecmp.clear_cache()
    assert filecmp.cmp(test, expected) is True
