"""
Convenience script to load all the different models
"""

from model.dbf import DBFLinear
from model.wls_models import StationaryLinear, WLSLinear, StationaryMean, ARIMA011, SlidingLinear, PowerLinear, HingeLinear
from model.statespace import StateSpace
from model.arima import Arima