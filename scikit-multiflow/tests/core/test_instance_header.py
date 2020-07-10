from skmultiflow.core.instance_header import InstanceHeader


def test_instance_header():
    header = InstanceHeader(header=['foo', 'bar', 'target'])

    assert header.get_info() == "InstanceHeader: header: ['foo', 'bar', 'target']"

    assert header.get_header_label_at(0) == 'foo'

    assert header.get_header_label_at(4) is None
