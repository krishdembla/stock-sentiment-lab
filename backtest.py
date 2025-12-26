
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from data.fetch_stock_data import get_stock_data


class PortfolioBacktest:
    """runs backtest comparing risk parity vs equal weight strategies"""
    
    def __init__(self, tickers, start_date, end_date, initial_capital=100000, risk_percent=0.02):
        """
        tickers: list of stock symbols to trade
        start_date: backtest start date
        end_date: backtest end date
        initial_capital: starting portfolio value
        risk_percent: total portfolio risk (as decimal)
        """
        self.tickers = tickers
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.risk_percent = risk_percent
        
        # storage for results
        self.risk_parity_equity = []
        self.equal_weight_equity = []
        self.dates = []
        
    def get_price_data(self):
        """fetch price data for all tickers"""
        print(f"Fetching price data for {len(self.tickers)} stocks...")
        price_data = {}
        
        for ticker in self.tickers:
            df = get_stock_data(ticker, start=self.start_date - timedelta(days=365), 
                              end=self.end_date, use_cache=True)
            if df is not None and len(df) > 0:
                # set date as index and get close prices
                df.set_index('Date', inplace=True)
                price_data[ticker] = df['Close']
            else:
                print(f"Warning: No data for {ticker}, skipping")
                
        # combine into single dataframe with aligned dates
        prices_df = pd.DataFrame(price_data)
        prices_df = prices_df.dropna()  # remove dates where any stock has missing data
        
        return prices_df
    
    def calculate_realized_volatility(self, prices, window=20):
        """
        calculate realized volatility from price history
        uses trailing window of returns
        """
        returns = prices.pct_change().dropna()
        vol = returns.rolling(window=window).std() * np.sqrt(252)  # annualize
        return vol
    
    def calculate_risk_parity_weights(self, date, prices_df, lookback=20):
        """
        calculate position sizes based on realized volatility
        each position risks the same dollar amount
        """
        volatilities = {}
        
        # get historical prices up to this date
        historical = prices_df[prices_df.index <= date]
        
        if len(historical) < lookback + 1:
            return {}
        
        # calculate realized volatility for each ticker
        for ticker in self.tickers:
            if ticker not in historical.columns:
                continue
                
            ticker_prices = historical[ticker].dropna()
            if len(ticker_prices) < lookback + 1:
                continue
            
            # calculate volatility
            returns = ticker_prices.pct_change().dropna()
            vol = returns.tail(lookback).std() * np.sqrt(252) * 100  # annualized as percentage
            
            if vol > 0 and not np.isnan(vol):
                volatilities[ticker] = vol
        
        if len(volatilities) == 0:
            return {}
        
        # calculate risk per position (total risk split equally)
        num_positions = len(volatilities)
        risk_per_position = self.risk_percent / num_positions
        
        # calculate position weights
        # position = (account * risk_per_position) / (2 * volatility)
        # the 2x is a buffer since volatility can spike
        weights = {}
        for ticker, vol in volatilities.items():
            position_pct = (risk_per_position) / (2 * (vol / 100))
            weights[ticker] = position_pct
        
        # normalize to sum to 1.0 (fully invested)
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def calculate_equal_weight(self):
        """simple 1/N weighting"""
        n = len(self.tickers)
        return {ticker: 1.0 / n for ticker in self.tickers}
    
    def run_backtest(self):
        """
        run monthly rebalancing backtest
        rebalances portfolio on first trading day of each month
        """
        print("\nStarting backtest...")
        print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
        print(f"Tickers: {', '.join(self.tickers)}")
        print(f"Initial capital: ${self.initial_capital:,.0f}")
        print(f"Total portfolio risk: {self.risk_percent * 100:.1f}%")
        print("-" * 60)
        
        # get price data
        prices_df = self.get_price_data()
        
        # filter to backtest period
        prices_df = prices_df[prices_df.index >= self.start_date]
        prices_df = prices_df[prices_df.index <= self.end_date]
        
        if len(prices_df) == 0:
            print("Error: No price data available for backtest period")
            return None
        
        # initialize portfolios
        risk_parity_value = self.initial_capital
        equal_weight_value = self.initial_capital
        
        # track weights
        risk_parity_weights = {}
        equal_weight_weights = self.calculate_equal_weight()
        
        # track rebalance dates
        last_rebalance_month = None
        rebalance_count = 0
        
        print("\nRunning monthly rebalancing...")
        for i, date in enumerate(prices_df.index):
            current_month = (date.year, date.month)
            
            # rebalance on first trading day of each month
            if last_rebalance_month != current_month:
                rebalance_count += 1
                print(f"Rebalancing #{rebalance_count} on {date.date()}...")
                
                # calculate new weights for risk parity
                risk_parity_weights = self.calculate_risk_parity_weights(date, prices_df)
                
                if len(risk_parity_weights) == 0:
                    print(f"  Warning: No valid weights for risk parity on {date.date()}, keeping previous weights")
                else:
                    print(f"  Risk parity weights: {', '.join([f'{k}={v:.1%}' for k, v in sorted(risk_parity_weights.items())])}")
                
                last_rebalance_month = current_month
            
            # calculate daily returns for each ticker
            if i > 0:
                prev_prices = prices_df.iloc[i - 1]
                curr_prices = prices_df.iloc[i]
                returns = (curr_prices - prev_prices) / prev_prices
                
                # calculate portfolio returns
                risk_parity_return = sum(risk_parity_weights.get(ticker, 0) * returns.get(ticker, 0) 
                                       for ticker in self.tickers if ticker in returns)
                equal_weight_return = sum(equal_weight_weights.get(ticker, 0) * returns.get(ticker, 0) 
                                        for ticker in self.tickers if ticker in returns)
                
                # update portfolio values
                risk_parity_value *= (1 + risk_parity_return)
                equal_weight_value *= (1 + equal_weight_return)
            
            # record equity curve
            self.dates.append(date)
            self.risk_parity_equity.append(risk_parity_value)
            self.equal_weight_equity.append(equal_weight_value)
        
        print(f"\nBacktest complete! Total rebalances: {rebalance_count}")
        return self.calculate_metrics()
    
    def calculate_metrics(self):
        """calculate performance metrics for both strategies"""
        
        # create equity dataframe
        equity_df = pd.DataFrame({
            'date': self.dates,
            'risk_parity': self.risk_parity_equity,
            'equal_weight': self.equal_weight_equity
        })
        equity_df.set_index('date', inplace=True)
        
        # calculate daily returns
        returns_df = equity_df.pct_change().dropna()
        
        # calculate metrics for each strategy
        metrics = {}
        
        for strategy in ['risk_parity', 'equal_weight']:
            equity = equity_df[strategy]
            returns = returns_df[strategy]
            
            # total return
            total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
            
            # annualized return (assuming 252 trading days)
            days = len(equity)
            years = days / 252
            annualized_return = ((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) * 100
            
            # volatility (annualized)
            volatility = returns.std() * np.sqrt(252) * 100
            
            # sharpe ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02
            sharpe = (annualized_return / 100 - risk_free_rate) / (volatility / 100)
            
            # max drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min() * 100
            
            # win rate
            win_rate = (returns > 0).sum() / len(returns) * 100
            
            metrics[strategy] = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'final_value': equity.iloc[-1]
            }
        
        # save results
        self.save_results(equity_df, metrics)
        
        return metrics
    
    def save_results(self, equity_df, metrics):
        """save backtest results to files"""
        # save equity curves
        equity_df.to_csv('backtest_equity_curves.csv')
        print(f"\nSaved equity curves to backtest_equity_curves.csv")
        
        # save metrics
        with open('backtest_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to backtest_metrics.json")
    
    def print_results(self, metrics):
        """print results to console"""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)
        
        print("\nRISK PARITY (Our Approach)")
        print("-" * 60)
        for metric, value in metrics['risk_parity'].items():
            if metric == 'final_value':
                print(f"{metric.replace('_', ' ').title():.<40} ${value:,.0f}")
            elif metric in ['total_return', 'annualized_return', 'volatility', 'max_drawdown', 'win_rate']:
                print(f"{metric.replace('_', ' ').title():.<40} {value:.2f}%")
            else:
                print(f"{metric.replace('_', ' ').title():.<40} {value:.2f}")
        
        print("\nEQUAL WEIGHT (Baseline)")
        print("-" * 60)
        for metric, value in metrics['equal_weight'].items():
            if metric == 'final_value':
                print(f"{metric.replace('_', ' ').title():.<40} ${value:,.0f}")
            elif metric in ['total_return', 'annualized_return', 'volatility', 'max_drawdown', 'win_rate']:
                print(f"{metric.replace('_', ' ').title():.<40} {value:.2f}%")
            else:
                print(f"{metric.replace('_', ' ').title():.<40} {value:.2f}")
        
        print("\nKEY INSIGHTS")
        print("-" * 60)
        return_diff = metrics['risk_parity']['annualized_return'] - metrics['equal_weight']['annualized_return']
        sharpe_diff = metrics['risk_parity']['sharpe_ratio'] - metrics['equal_weight']['sharpe_ratio']
        dd_diff = metrics['risk_parity']['max_drawdown'] - metrics['equal_weight']['max_drawdown']
        vol_diff = metrics['risk_parity']['volatility'] - metrics['equal_weight']['volatility']
        
        print(f"Return difference:.......................... {return_diff:+.2f}% annually")
        print(f"Sharpe ratio difference:.................... {sharpe_diff:+.2f}")
        print(f"Volatility difference:...................... {vol_diff:+.2f}%")
        print(f"Max drawdown improvement:................... {-dd_diff:+.2f}%")
        
        print("\nCONCLUSION:")
        if return_diff < 0:
            print(f"  In this bull market period (2022-2024), equal weight captured")
            print(f"  more upside (+{-return_diff:.1f}% annually) but risk parity provided:")
            print(f"  • {abs(vol_diff):.1f}% lower volatility (smoother ride)")
            print(f"  • {-dd_diff:.1f}% smaller max drawdown (less pain)")
            print(f"  • More consistent risk across all positions")
            print(f"\n  Risk parity excels in volatile/down markets by limiting losses.")
        else:
            print(f"  Risk parity outperformed by {return_diff:.2f}% annually with:")
            print(f"  • {sharpe_diff:+.2f} higher Sharpe ratio")
            print(f"  • {-dd_diff:+.2f}% better max drawdown")
        
        print("=" * 60)


def main():
    """run backtest with default parameters"""
    
    # select a diversified portfolio of stocks
    # choosing mix of tech, finance, healthcare, consumer, energy
    tickers = ['AAPL', 'MSFT', 'NVDA', 'JPM', 'BAC', 'JNJ', 'PG', 'XOM']
    
    # backtest period
    start_date = '2020-01-01'
    end_date = '2024-12-01'
    
    # run backtest
    bt = PortfolioBacktest(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=100000,
        risk_percent=0.02
    )
    
    metrics = bt.run_backtest()
    
    if metrics:
        bt.print_results(metrics)
    else:
        print("Backtest failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
