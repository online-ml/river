"""Datasets."""
from .airline import Airline
from .chick_weights import ChickWeights
from .credit_card import CreditCard
from .elec2 import Elec2
from .kdd99_http import KDD99HTTP
from .phishing import Phishing
from .restaurants import Restaurants
from .sms import SMS
from .toulouse_bikes import ToulouseBikes
from .trec07 import TREC07
from .trump_approval import TrumpApproval


__all__ = [
    'Airline',
    'ChickWeights',
    'CreditCard',
    'Elec2',
    'KDD99HTTP',
    'Phishing',
    'Restaurants',
    'SMS',
    'ToulouseBikes',
    'TREC07',
    'TrumpApproval'
]
