# --- coding: utf-8 ---
from freqtrade.strategy import IStrategy
import numpy as np
import pandas as pd

class SwingAnchoredVwapBigBeluga(IStrategy):
    # Estratégia apenas para fins não comerciais — atente para a licença
    
    # Configurações
    minimal_roi = {
        "0": 0.10
    }
    stoploss = -0.15
    timeframe = '1h'
    trailing_stop = True
    trailing_stop_positive = 0.01

    # Parâmetro VWAP
    length = 50

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Identificar swing high / low
        dataframe['swing_high'] = dataframe['high'].rolling(window=self.length, min_periods=1).max()
        dataframe['swing_low'] = dataframe['low'].rolling(window=self.length, min_periods=1).min()

        # Ancorar VWAP nos swings
        dataframe['vwap_high'] = self.calc_vwap(dataframe, 'high', self.length)
        dataframe['vwap_low'] = self.calc_vwap(dataframe, 'low', self.length)
        
        # Determinar tendência - bull se novo high, bear se novo low
        dataframe['trend'] = np.where(dataframe['high'] == dataframe['swing_high'], True,
                                      np.where(dataframe['low'] == dataframe['swing_low'], False, np.nan))
        dataframe['trend'] = dataframe['trend'].ffill()

        return dataframe

    def calc_vwap(self, dataframe, anchor, lookback):
        rolls = []
        prices = dataframe[anchor]
        volumes = dataframe['volume']
        for i in range(len(prices)):
            start = max(0, i-lookback+1)
            vwap = np.sum(prices[start:i+1] * volumes[start:i+1]) / np.sum(volumes[start:i+1] if np.sum(volumes[start:i+1]) != 0 else 1)
            rolls.append(vwap)
        return pd.Series(rolls, index=dataframe.index)

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Sinal de compra: tendência bull e fechamento > VWAP ancorado no swing low
        dataframe.loc[
            (dataframe['trend'] == True) &
            (dataframe['close'] > dataframe['vwap_low']),
            'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        # Sinal de venda: tendência bear e fechamento < VWAP ancorado no swing high
        dataframe.loc[
            (dataframe['trend'] == False) &
            (dataframe['close'] < dataframe['vwap_high']),
            'sell'] = 1
        return dataframe
