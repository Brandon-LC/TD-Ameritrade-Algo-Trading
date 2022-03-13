from tda import auth, client
from tda.orders.common import Duration, Session
from tda.orders.equities import equity_buy_limit
import json
import config
import datetime
import csv

# Authenticates TD Ameritrade API and returns a token of authentication.
try: # Initializes client with token.
    c = auth.client_from_token_file(config.token_path, config.api_key)
except FileNotFoundError: # If there is no token, then create one.
    from selenium import webdriver
    with webdriver.Chrome(executable_path=config.chromedriver_path) as driver: # Saves token in Chrome Driver folder. 
        # Use Chrome Driver to login to TD Ameritrade.
        c = auth.client_from_login_flow(
            driver, config.api_key, config.redirect_uri, config.token_path)

# Can place buy orders for a symbol. Specify the buy limit and number of shares.
def PlaceOrder(shares, symbol, limit):
    client.place_order(config.account_id,
        equity_buy_limit(symbol, shares, limit)
        .set_duration(Duration.GOOD_TILL_CANCEL)
        .set_session(Session.NORMAL)
        .build())

# Creates a CSV file with last 20 years of a stocks price history (Datetime, Open, High, Low, Close, Volume).
# Saves file to Stock History folder.
def GetData(symbols, location):
    for symbol in symbols: # Gets price history for each symbol.
        r = c.get_price_history(symbol,
                period_type=client.Client.PriceHistory.PeriodType.YEAR, # Type of period.
                period=client.Client.PriceHistory.Period.TWENTY_YEARS, # Number of periods.
                frequency_type=client.Client.PriceHistory.FrequencyType.DAILY, # Type of frequency for a new candle to be formed.
                frequency=client.Client.PriceHistory.Frequency.DAILY) # Number of frequencyType included in each candle.

        response = r.json() # Sets to json notation.
        dictionary = response['candles'] # Gets the stocks dictionary.

        with open(location + symbol + '.csv', 'w', newline='') as csvfile: # Creates and writes to a csv file.
            fieldnames = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume'] # The field names.
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames) # Creates a csv file.
            writer.writeheader() # Writes the field names.

            for candle in dictionary: # For each day, add it to the csv file in a new row.
                writer.writerow({'Datetime': datetime.datetime.fromtimestamp(int(candle['datetime'])/1000).strftime('%Y-%m-%d'), # Need to divide epoch time by 1000 to get rid of milliseconds.
                            'Open':candle['open'], 'High':candle['high'],
                            'Low':candle['low'], 'Close':candle['close'],
                            'Volume':candle['volume']})          

location = 'C:\\Users\\blcsi\\Desktop\\Algo\\Stock History\\' # Location to save CSV file to.
GetData(['AAPL','BA'], location) # Get stock price history for APPL and BA.