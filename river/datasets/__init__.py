"""Datasets.

This module contains a collection of datasets for multiple tasks: classification, regression, etc.
The data corresponds to popular datasets and are conveniently wrapped to easily iterate over
the data in a stream fashion. All datasets have fixed size. Please refer to `river.synth` if you
are interested in infinite synthetic data generators.

"""
from .airline_passengers import AirlinePassengers
from .bananas import Bananas
from .bikes import Bikes
from .chick_weights import ChickWeights
from .credit_card import CreditCard
from .elec2 import Elec2
from .higgs import Higgs
from .http import HTTP
from .insects import Insects
from .malicious_url import MaliciousURL
from .music import Music
from .movielens100k import MovieLens100K
from .phishing import Phishing
from .restaurants import Restaurants
from .segment import ImageSegments
from .sms_spam import SMSSpam
from .smtp import SMTP
from .solar_flare import SolarFlare
from .taxis import Taxis
from .trec07 import TREC07
from .trump_approval import TrumpApproval


__all__ = [
    "AirlinePassengers",
    "Bananas",
    "Bikes",
    "ChickWeights",
    "CreditCard",
    "Elec2",
    "Higgs",
    "HTTP",
    "ImageSegments",
    "Insects",
    "MaliciousURL",
    "MovieLens100K",
    "Music",
    "Phishing",
    "Restaurants",
    "SMSSpam",
    "SMTP",
    "SolarFlare",
    "Taxis",
    "TREC07",
    "TrumpApproval",
]
