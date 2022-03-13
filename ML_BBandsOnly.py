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

    # Defines classifier with features.
    def DefineClassifier(data):
        close = data.Close.values # Closing values.
        upper, lower = BBANDS(data, 20, 2) # 20 day moving avg, 2 std deviations.

        # Indicator features
        data['X_BB_upper'] = (upper - close) / close # Upper Bollinger Bands band.
        data['X_BB_lower'] = (lower - close) / close # Lower Bollinger Bands band.
        data['X_BB_width'] = (upper - lower) / close # Difference between Bollinger Bands bands.

        # DateTime Features.
        data['X_day'] = data.index.dayofweek # Day.

        data = data.dropna().astype(float) # Removes any rows with NAN values.
        return data

    # Tests the classification accuracies using KNN, Perceptron, RandomForest, and DecisionTree.
    # Adds the results to a dictionary.
    def ClassificationAccuracy():
        X, y = get_clean_Xy(data) # Get model and labels.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.5, random_state=0) # Creates testing and training data sets.

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

    data = DefineClassifier(data) # Defines classifier.

    classifiers = ClassificationAccuracy() # Prints classification accuracies.

    N_TRAIN = int(.2 * len(data.count(axis='columns'))) # Number of training examples is 20% of data.

    print("")
    print("MLTrainOnceStrategy: ")
    bt = Backtest(data, MLTrainOnceStrategy) # Run backtest using no updates.
    print(bt.run())
    bt.plot()

    #print("")
    #print("MLWalkForwardStrategy: ")
    #bt = Backtest(data, MLWalkForwardStrategy) # Run backtest updating every 20 iterations.
    #print(bt.run())
    #bt.plot()

Run()