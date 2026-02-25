#!/usr/bin/env python3
"""技术面 Alpha 因子库（量价类）"""
import numpy as np
import pandas as pd


def momentum_factor(close: pd.Series, window: int = 20) -> pd.Series:
    return close.pct_change(window)


def volatility_factor(close: pd.Series, window: int = 20) -> pd.Series:
    return -close.pct_change().rolling(window).std()


def volume_momentum(volume: pd.Series, window: int = 20) -> pd.Series:
    return volume / volume.rolling(window).mean()


def price_volume_corr(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    return close.rolling(window).corr(volume)