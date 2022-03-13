import pandas as pd
import numpy as np
from backtesting.test import SMA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from backtesting import Backtest, Strategy

def Run():
    # Get CSV file with 20 years of AAPL price history (Datetime, Open, High, Low, Close, Volume).
    def get_file(file):
        return pd.read_csv(file, index_col=0, parse_dates=True, infer_datetime_format=True)

    # Creates the upper and lower Bollinger Bands.
    def BBANDS(data, n_lookback, n_std):
        """Bollinger bands indicator"""
        hlc3 = (data.High + data.Low + data.Close) / 3 # Typical price.
        mean, std = hlc3.rolling(n_lookback).mean(), hlc3.rolling(n_lookback).std() # Finds statistics using pandas rolling frames.
        upper = mean + n_std*std # Upper Bollinger Band.
        lower = mean - n_std*std # Lower Bollinger Band.
        return upper, lower

    # Finds the Average Daily Trading Volume.
    def ADTV(data, n_lookback):
        return pd.Series(data.Volume).rolling(n_lookback).mean()

    # Finds the Standard Deviation.
    def STD(data):
        return pd.Series(data.Close).std()
    
    # Finds the Stochastic Oscillator.
    def Stochastic(data):
        high14 = pd.Series(data.High).rolling(14).max() # 14 day trading high.
        low14 = pd.Series(data.Low).rolling(14).min() # 14 day trading low.
        K = 100 * ((data.Close - low14) / (high14 - low14)) # Slow Stochastic indicator.

        high3 = pd.Series(data.High).rolling(3).max() # 3 day trading high.
        low3 = pd.Series(data.Low).rolling(3).min() # 3 day trading low.
        D = 100 * (high3 / low3) # Fast stochastic indicator.
        return K, D
    
    # Conversion Line (Tenkan-Sen) for Ichimoku Cloud.
    def ConversionLine(data):
        low9 = pd.Series(data.Low).rolling(9).min() # 9 day trading low.
        high9 = pd.Series(data.High).rolling(9).max() # 9 day trading high.
        return (low9 + high9) / 2

    # Base Line (Kijun Sen) for Ichimoku Cloud.
    def BaseLine(data):
        high26 = pd.Series(data.High).rolling(26).max() # 26 day trading high.
        low26 = pd.Series(data.Low).rolling(26).min() # 26 day trading low.
        return 0.5 * (high26 + low26)
    
    # Finds the Ichimoku Cloud.
    def IchimokuCloud(data):
        conversionLine = ConversionLine(data)
        baseLine = BaseLine(data)
        LeadingSpanA = (conversionLine + baseLine) / 2 # Senkou Span A.
        high52 = pd.Series(data.High).rolling(52).max() # 52 day trading high.
        low52 = pd.Series(data.High).rolling(52).min() # 52 day trading low.
        LeadingSpanB = (high52 + low52) / 2 # Senkou Span B.

        return LeadingSpanA - LeadingSpanB
    
    # Finds the Exponential Moving Average for Moving Average Convergence Divergence.
    def EMA(data, n_lookback):
        multiplier = 2 / (n_lookback + 1) # Gets multiplier.
        firstEMA = pd.Series(data.Close).rolling(n_lookback).sum() # Get first EMA.
        EMA = (data.Close * multiplier) + (firstEMA * (1 - multiplier)) # Gets EMA using pervious EMA.
        return EMA

    # Finds the Moving Average Convergence Divergence.
    def MACD(data):
        ema26 = EMA(data, 26) # 26 day Exponential Moving Average.
        ema12 = EMA(data, 12) # 12 day Exponential Moving Average.
        return ema12 - ema26

    # Returns only columns containing 'X' in their name (model features).
    def get_X(data):
        return data.filter(like='X').values

    # Creates labels, (1, 0, -1).
    def get_y(data):
        y = data.Close.pct_change(2).shift(-2)  # Stock's close percent change with 2 day period, shift so the change will be placed in the correct row.
        y[y.between(-.004, .004)] = 0 # Devalue if smaller than 0.4% change.
        y[y > 0] = 1 # Stock went up.
        y[y < 0] = -1 # Stock went down.
        return y
    
    # Removes NaN values.
    def get_clean_Xy(df):
        X = get_X(df)
        y = get_y(df).values
        isnan = np.isnan(y)
        X = X[~isnan]
        y = y[~isnan]
        return X, y

    # Defines classifier with a bunch of features.
    def DefineClassifier(data):
        # Get a load of random features.
        upper, lower = BBANDS(data, 20, 2) # 20 day Bollinger Bands, with 20 day period and 2 std deviations.
        adtv = ADTV(data, 20) # 20 day Average Daily Training Volume.
        std = STD(data) # Standard Deviation of closing prices.
        K, D = Stochastic(data) # Fast and slow moving Stochastic indicators.
        ichimoku = IchimokuCloud(data) # Ichimoku Cloud.
        macd = MACD(data) # Moving Average Convergence Divergence
        volume = data.Volume # Daily volume.
        close = data.Close.values # Closing values.

        sma10 = SMA(data.Close, 10) # 10 day Simple Moving Avg, using pandas rolling window.
        sma20 = SMA(data.Close, 20) # 20 day Simple Moving Avg
        sma50 = SMA(data.Close, 50) # 50 day Simple Moving Avg
        sma100 = SMA(data.Close, 100) # 100 day Simple Moving Avg
        sma250 = SMA(data.Close, 250) # 250 day Simple Moving Avg

        # Price-derived features
        data['X_SMA10'] = (close - sma10) / close # Close price in relation to 10 day SMA.
        data['X_SMA20'] = (close - sma20) / close # Close price in relation to 20 day SMA.
        data['X_SMA50'] = (close - sma50) / close # Close price in relation to 50 day SMA.
        data['X_SMA100'] = (close - sma100) / close # Close price in relation to 100 day SMA.
        data['X_SMA250'] = (close - sma250) / close # Close price in relation to 250 day SMA.

        data['X_DELTA_SMA10'] = (sma10 - sma20) / close # 10 day SMA in relation to 20 day SMA.
        data['X_DELTA_SMA20'] = (sma20 - sma50) / close # 20 day SMA in relation to 50 day SMA.
        data['X_DELTA_SMA50'] = (sma50 - sma100) / close # 50 day SMA in relation to 100 day SMA.
        data['X_DELTA_SMA100'] = (sma100 - sma250) / close # 100 day SMA in relation to 250 day SMA.

        # Indicator features
        data['X_MOM'] = data.Close.pct_change(periods=2) # Close percent change with a 2 day period.

        data['X_BB_upper'] = (upper - close) / close # Upper Bollinger Bands band.
        data['X_BB_lower'] = (lower - close) / close # Lower Bollinger Bands band.
        data['X_BB_width'] = (upper - lower) / close # Difference between upper and lower Bollinger Bands bands.
        
        data['X_ADTV'] = adtv # Avg volume.
        data['X_STD'] = std # Standard deviation.

        data['X_DK_fast'] = (K - close) / close # Fast Stochastic.
        data['X_DK_slow'] = (D - close) / close # Slow Stochastic.
        data['X_DK_width'] = (K - D) / close # Difference between Stochastic indicators.

        data['X_CLOUD'] = ichimoku # Ichimoku cloud.
        data['X_MACD'] = macd # MACD.
        data['X_VOL'] = volume # Daily volume.
        
        # Datetime features.
        data['X_day'] = data.index.dayofweek # Day.

        data = data.dropna().astype(float) # Removes any rows with NAN values.
        return data

    # Tests the classification accuracies using KNN, Perceptron, Linear-SVC, RandomForest, and DecisionTree.
    # Adds the results to a dictionary.
    def ClassificationAccuracy():
        X, y = get_clean_Xy(data) # Get model and labels.
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0) # Creates testing and training data sets.

        clfKNN = KNeighborsClassifier(7)  # Model the output based on 7 "nearest" examples.
        clfP = Perceptron()
        clfRF = RandomForestClassifier(random_state=0)
        clfDC = DecisionTreeClassifier(random_state=0) 

        clfKNN.fit(X_train, y_train) # Classifier is fit to the training data.
        clfP.fit(X_train, y_train)
        clfRF.fit(X_train, y_train)
        clfDC.fit(X_train, y_train)

        y_predKNN = clfKNN.predict(X_test) # Predict the class labels using the test data.
        y_predP = clfP.predict(X_test)
        y_predRF = clfRF.predict(X_test)
        y_predDC = clfDC.predict(X_test)

        # Dictionary containing all classifiers and their accuracies.
        classifiers = {KNeighborsClassifier(7) : np.mean(y_test == y_predKNN),
                Perceptron() : np.mean(y_test == y_predP),
                RandomForestClassifier(random_state=0) : np.mean(y_test == y_predRF),
                DecisionTreeClassifier(random_state=0) : np.mean(y_test == y_predDC)}

        print('Classification accuracy K-Nearest Neighbor: ', np.mean(y_test == y_predKNN)) # Calculates the mean of the classifiers correct predictions.
        print('Classification accuracy Perceptron: ', np.mean(y_test == y_predP))
        print('Classification accuracy Random Forest: ', np.mean(y_test == y_predRF))
        print('Classification accuracy Decision Tree: ', np.mean(y_test == y_predDC))

        return classifiers

    # Buys the asset for 20% of available equity whenever the forecast is positive, sells when negative, while setting reasonable stop-loss and take-profit levels.
    class MLTrainOnceStrategy(Strategy):
        price_delta = .004  # 0.4%, used for stop-loss and take-profit levels.

        # Init our model using the most accurate classifier.
        def init(self):        
            self.clf = max(classifiers, key=classifiers.get) # Uses the most accurate classifier.

            # Train the classifier in advance on the first N_TRAIN examples
            df = self.data.df.iloc[:N_TRAIN]
            X, y = get_clean_Xy(df) # Create labels using training data (N_TRAIN).
            self.clf.fit(X, y) # Fit classifier to labels.

            # Plot y for inspection
            self.I(get_y, self.data.df, name='y_true')

            # Prepare empty, all-NaN forecast indicator
            self.forecasts = self.I(lambda: np.repeat(np.nan, len(self.data)), name='forecast')

        # Runs algorithm on each new candle stick update.
        def next(self):
            if len(self.data) < N_TRAIN: # Skip the training, in-sample data
                return

            # Proceed only with out-of-sample data. Prepare some variables
            high, low, close = self.data.High, self.data.Low, self.data.Close
            current_time = self.data.index[-1]

            # Forecast the next movement
            X = get_X(self.data.df.iloc[-1:]) # Gets last class label.
            forecast = self.clf.predict(X)[0] # Gets last prediction.

            # Update the plotted "forecast" indicator
            self.forecasts[-1] = forecast

            # If our forecast is upwards and we don't already hold a long position place a long order for 20% of available account equity. Vice versa for short.
            # Also set target take-profit and stop-loss prices to be one price_delta away from the current closing price.
            upper, lower = close[-1] * (1 + np.r_[1, -1]*self.price_delta) # Calculates price_delta.

            if forecast == 1 and not self.position.is_long: # Predicts stock will go up.
                self.buy(size=.2, tp=upper, sl=lower)
            elif forecast == -1 and not self.position.is_short: # Predicts stock will go down.
                self.sell(size=.2, tp=lower, sl=upper)

            # Additionally, set aggressive stop-loss on trades that have been open for more than two days
            for trade in self.trades:
                if current_time - trade.entry_time > pd.Timedelta('2 days'):
                    if trade.is_long: # In a long position.
                        trade.sl = max(trade.sl, low) # Increases stop-loss to low.
                    else: # In short position.
                        trade.sl = min(trade.sl, high) # Decrease stop-loss to high.

    # How does it perform under walk-forward optimization, akin to k-fold or leave-one-out cross-validation (using 20 iterations).
    class MLWalkForwardStrategy(MLTrainOnceStrategy):
        def next(self):
            if len(self.data) < N_TRAIN: # Skip the cold start period with too few values available
                return

            # Re-train the model only every 20 iterations.
            # Since 20 << N_TRAIN, we don't lose much in terms of "recent training examples", but the speed-up is significant!
            if len(self.data) % 20:
                return super().next()

            # Retrain on last N_TRAIN values
            df = self.data.df[-N_TRAIN:]
            X, y = get_clean_Xy(df)
            self.clf.fit(X, y)

            # Now that the model is fitted, proceed the same as in MLTrainOnceStrategy
            super().next()

    AAPL = get_file('C:\\Users\\blcsi\\Desktop\\Algo\\Stock History\\AAPL.csv')
    data = AAPL.copy() # AAPL price history.

    data = DefineClassifier(data) # Defines classifier with a load of random features.

    classifiers = ClassificationAccuracy() # Prints classification accuracies.
    
    N_TRAIN = int(.2 * len(data.count(axis='columns'))) # Number of training examples is 20% of data.

    print("")
    print("MLTrainOnceStrategy: ")
    bt = Backtest(data, MLTrainOnceStrategy) # Run backtest using no updates.
    print(bt.run())
    #bt.plot() # Prints plot of stats.

    print("")
    print("MLWalkForwardStrategy: ")
    bt = Backtest(data, MLWalkForwardStrategy) # Run backtest updating every 20 iterations.
    print(bt.run())
    #bt.plot() # Prints plot of stats.

Run()