"""Datasets."""
from . import gen
from .airline import Airline
from .bananas import Bananas
from .chick_weights import ChickWeights
from .credit_card import CreditCard
from .elec2 import Elec2
from .higgs import Higgs
from .http import HTTP
from .phishing import Phishing
from .movielens100k import MovieLens100K
from .restaurants import Restaurants
from .segment import ImageSegments
from .sms_spam import SMSSpam
from .smtp import SMTP
from .bikes import Bikes
from .taxis import Taxis
from .trec07 import TREC07
from .trump_approval import TrumpApproval
from .malicious_url import MaliciousURL


__all__ = [
    'Airline',
    'Bananas',
    'Bikes',
    'ChickWeights',
    'CreditCard',
    'Elec2',
    'gen',
    'Higgs',
    'HTTP',
    'MaliciousURL',
    'Phishing',
    'MovieLens100K',
    'Restaurants',
    'ImageSegments',
    'SMSSpam',
    'SMTP',
    'Taxis',
    'TREC07',
    'TrumpApproval'
]
