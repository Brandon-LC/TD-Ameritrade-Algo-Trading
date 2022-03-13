import pandas as pd
import numpy as np
from backtesting.test import SMA
from backtesting import Backtest, Strategy

def Run():
    # Get CSV file with 20 years of AAPL price history (Datetime, Open, High, Low, Close, Volume).
    def get_file(file):
            return pd.read_csv(file,
                    index_col=0, parse_dates=True, infer_datetime_format=True)

    # Returns the data as a pandas series.
    def test(data, n):
        return pd.Series(data)

    # Optimizes the variables.
    def Optimize():
        stats = bt.optimize(price=range(1, 10, 1), # Stop-loss and target price percentage.
                        SMADays=range(10, 250, 10), # SMA day average.
                        percent=range(1, 10, 1), # Percent loss to trigger a buy order.
                        maximize='Equity Final [$]') # Maximize final equity.
        print(stats._strategy)

    # Gap reversal strategy using default paramaters.
    class GapReversal(Strategy):
        price_delta = .004 # .4% stop-loss and target price.
        SMADays = 250 # 250 day SMA.
        percentLoss = .01 # 1% gap down.

        # Initializes default values.
        def init(self):       
            self.open = self.I(test, self.data.Open , 0)
            self.close = self.I(test, self.data.Close, 0)
            self.high = self.I(test, self.data.High, 0)
            self.low = self.I(test, self.data.Low, 0)
            self.sma250 = self.I(SMA, self.data.Close, self.SMADays)

        # Runs strategy on each new candle stick.
        def next(self):
            upper, lower = self.close[-1] * (1 + np.r_[1, -1]*self.price_delta) # Gets the stop-loss and target price.
            current_time = self.data.index[-1] # Gets the stocks date.

            if ((self.open[-1] - self.close[-2])/self.close[-2] <= - self.percentLoss and # Is the gap down more then 1%.
            self.close[-1] > self.open[-1] and # Is the close price higher then the open.
            self.close[-1] > ((self.high[-1] - self.low[-1])/2) + self.low[-1] and # Is the close price higher then the mid-price.
            self.close[-1] > self.sma250[-1]): # Is the close price higher then the 250 day SMA.
                self.buy(size=.2, tp=upper, sl=lower) # Then buy with 20% of avaliable equity.

            # Additionally, set aggressive stop-loss on trades that have been open for more than two days
            for trade in self.trades:
                if current_time - trade.entry_time > pd.Timedelta('2 days'):
                    if trade.is_long: # In a long position.
                        trade.sl = max(trade.sl, low) # Increases stop-loss to low.
                    else: # In short position.
                        trade.sl = min(trade.sl, high) # Decrease stop-loss to high.

    # Gap reversal strategy using optimized paramaters.
    class GapReversalOptimal(Strategy):
        price = 1
        price_delta = price * .001 # .01% stop-loss and target price.

        SMADays = 10 # 10 day SMA.

        percent = 1
        percentLoss = percent * .01 # Percent that counts as a gap.

        # Initializes default values.
        def init(self):       
            self.open = self.I(test, self.data.Open , 0)
            self.close = self.I(test, self.data.Close, 0)
            self.high = self.I(test, self.data.High, 0)
            self.low = self.I(test, self.data.Low, 0)
            self.sma250 = self.I(SMA, self.data.Close, self.SMADays)

        def next(self):
            upper, lower = self.close[-1] * (1 + np.r_[1, -1]*self.price_delta) # Gets the stop-loss and target price.
            current_time = self.data.index[-1] # Gets the stocks date.
            
            if ((self.open[-1] - self.close[-2])/self.close[-2] <= - self.percentLoss and # Is the gap down more then 1%.
            self.close[-1] > self.open[-1] and # Is the close price higher then the open.
            self.close[-1] > ((self.high[-1] - self.low[-1])/2) + self.low[-1] and # Is the close price higher then the mid-price.
            self.close[-1] > self.sma250[-1]): # Is the close price higher then the 250 day SMA.
                self.buy(size=.2, tp=upper, sl=lower) # Then buy with 20% of avaliable equity.

            # Additionally, set aggressive stop-loss on trades that have been open for more than two days
            for trade in self.trades:
                if current_time - trade.entry_time > pd.Timedelta('2 days'):
                    if trade.is_long: # In a long position.
                        trade.sl = max(trade.sl, low) # Increases stop-loss to low.
                    else: # In short position.
                        trade.sl = min(trade.sl, high) # Decrease stop-loss to high.

    AAPL = get_file('C:\\Users\\blcsi\\Desktop\\Algo\\Stock History\\AAPL.csv') # Gets APPL stock price history.

    bt = Backtest(AAPL, GapReversal, cash=10_000) # Runs Gap Reversal strategy.
    print(bt.run())
    #bt.plot()

    bt = Backtest(AAPL, GapReversalOptimal, cash=10_000) # Runs Optimal Gap Reversal strategy.
    #Optimize() # Finds optimal paramaters, need to manually set.
    print(bt.run())
    #bt.plot()

Run()