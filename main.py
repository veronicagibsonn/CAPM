from datetime import datetime, timedelta
from io import StringIO
from urllib import request, error, parse

import pandas as pd
import scipy as sp


def main():
    watchlist = {'MSFT', 'GOOG', 'TSLA', 'AMZN', 'VRTX',
                 'SPY', 'DIA', 'IWM', 'NSC', 'REXR', 'MEDP'}
    with open('snp500.txt', 'rt') as ticker_file:
        for line in ticker_file.readlines():
          watchlist.add(line[:-1])

    n_points = 50

    stats = []
    market_prices = search_yahoo('^GSPC', n_points)
    market_hpr = sp.signal.convolve(market_prices, [1, -1])[1:-1] / market_prices[:-1]
    regression = sp.stats.linregress(market_hpr, market_hpr)
    stats.append(['S&P 500', regression.intercept, 0, regression.slope, 0])
    rfs = search_yahoo('^IRX', n_points);
    rf = rfs.iloc[-1]
    daily_rf = (1 + rf / 100) ** (1 / 252) - 1
    market_hpr -= daily_rf

    for ticker in watchlist:
        try:
            prices = search_yahoo(ticker, n_points)
            hpr = sp.signal.convolve(prices, [1, -1])[1:-1] / prices[:-1]
            hpr -= daily_rf
        except ValueError as e:
            print(e)
            continue
        hpr.name = ticker
        regression = sp.stats.linregress(market_hpr, hpr)
        t_alpha = abs(regression.intercept / regression.intercept_stderr)
        p_alpha = 2 * sp.stats.t.cdf(-t_alpha, len(hpr) - 2)
        t_beta = abs((regression.slope - 1) / regression.stderr)
        p_beta = 2 * sp.stats.t.cdf(-t_beta, len(hpr) - 2)
        stats.append([ticker, regression.intercept, p_alpha, regression.slope, p_beta])

    stats = pd.DataFrame(stats, columns=['ticker', 'alpha', 'p(alpha=0)', 'beta', 'p(beta=1)'])
    stats = stats.sort_values('p(alpha=0)')
    print(stats.head(20))
    stats.to_csv('capm.csv', index=False)


def search_yahoo(ticker, n):
    base = 'https://query1.finance.yahoo.com/v7/finance/download/{}?period1={}' \
           '&period2={}&interval={}&events={}&includeAdjustedClose={}'
    t1 = datetime.now()
    t0 = t1 - timedelta(days=366)
    interval = '1d'  # one day
    events = 'history'  # price history
    adj = True  # include adjusted closing price

    try:
        response = request.urlopen(
            base.format(parse.quote(ticker), int(t0.timestamp()), int(t1.timestamp()),
                        interval, events, str(adj).lower()))
        if response.status != 200:
            raise ValueError(f'yahoo status {response.status} for ticker {ticker}')
    except error.HTTPError as e:
        raise ValueError(f'http error {e} for ticker {ticker}')
    csv = pd.read_csv(StringIO(response.read().decode()))
    if len(csv.index) < n:
        raise ValueError(f'too few values for ticker {ticker}')
    return csv.iloc[-n:]['Adj Close']


if __name__ == '__main__':
    main()
  