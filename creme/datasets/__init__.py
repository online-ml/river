"""Datasets."""
from . import gen
from .airline import Airline
from .banana import Banana
from .chick_weights import ChickWeights
from .credit_card import CreditCard
from .elec2 import Elec2
from .higgs import Higgs
from .kdd99_http import KDD99HTTP
from .phishing import Phishing
from .movielens100k import MovieLens100K
from .restaurants import Restaurants
from .sms_spam import SMSSpam
from .toulouse_bikes import ToulouseBikes
from .trec07 import TREC07
from .trump_approval import TrumpApproval
from .malicious_url import MaliciousURL


__all__ = [
    'Airline',
    'Banana',
    'ChickWeights',
    'CreditCard',
    'Elec2',
    'KDD99HTTP',
    'gen',
    'Higgs',
    'MaliciousURL',
    'Phishing',
    'MovieLens100K',
    'Restaurants',
    'SMSSpam',
    'ToulouseBikes',
    'TREC07',
    'TrumpApproval'
]
