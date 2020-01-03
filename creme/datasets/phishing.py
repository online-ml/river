import os

from .. import stream

from . import base


class Phishing(base.Dataset):
    """Phishing websites.

    This dataset contains features from web pages that are phishing websites or not.

    Yields:
        tuple: A pair (``x``, ``y``) where ``x`` is a dict of features and ``y`` is the target.

    References:
        1. `UCI page <https://archive.ics.uci.edu/ml/datasets/phishing+websites>`_

    """

    def __init__(self):
        super().__init__(
            n_samples=11055,
            n_features=30,
            category=base.BINARY_CLF
        )

    def _stream_X_y(self, directory):
        return stream.iter_csv(
            os.path.join(directory, 'phishing.csv.gz'),
            target_name='Result',
            converters={
                'having_IP_Address': float,
                'URL_Length': float,
                'Shortining_Service': float,
                'having_At_Symbol': float,
                'double_slash_redirecting': float,
                'Prefix_Suffix': float,
                'having_Sub_Domain': float,
                'SSLfinal_State': float,
                'Domain_registeration_length': float,
                'Favicon': float,
                'port': float,
                'HTTPS_token': float,
                'Request_URL': float,
                'URL_of_Anchor': float,
                'Links_in_tags': float,
                'SFH': float,
                'Submitting_to_email': float,
                'Abnormal_URL': float,
                'Redirect': float,
                'on_mouseover': float,
                'RightClick': float,
                'popUpWidnow': float,
                'Iframe': float,
                'age_of_domain': float,
                'DNSRecord': float,
                'web_traffic': float,
                'Page_Rank': float,
                'Google_Index': float,
                'Links_pointing_to_page': float,
                'Statistical_report': float,
                'Result': float
            }
        )
