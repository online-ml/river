"""Datasets."""
from . import synth
from .airline_passengers import AirlinePassengers
from .bananas import Bananas
from .chick_weights import ChickWeights
from .credit_card import CreditCard
from .elec2 import Elec2
from .higgs import Higgs
from .http import HTTP
from .insects import Insects
from .music import Music
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
    'AirlinePassengers',
    'Bananas',
    'Bikes',
    'ChickWeights',
    'CreditCard',
    'Elec2',
    'Higgs',
    'HTTP',
    'ImageSegments',
    'Insects',
    'MaliciousURL',
    'MovieLens100K',
    'Music',
    'Phishing',
    'Restaurants',
    'SMSSpam',
    'SMTP',
    'synth',
    'Taxis',
    'TREC07',
    'TrumpApproval'
]
