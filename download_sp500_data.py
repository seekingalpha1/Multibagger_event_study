"""
Script to download historical price data for all historical S&P 500 members
to avoid survivorship bias.

Features:
- Download historical data
- Load existing data
- Update existing data with new data
- Overwrite historical data if corrections were made
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import os
from tqdm import tqdm
import warnings
import logging

warnings.filterwarnings('ignore')
logging.getLogger('yfinance').disabled = True


class SP500DataManager:
    """
    Manager class for S&P 500 historical data.
    """
    
    def __init__(self, 
                 components_url="https://raw.githubusercontent.com/fja05680/sp500/master/S%26P%20500%20Historical%20Components%20%26%20Changes%2801-17-2026%29.csv",
                 data_file='sp500_historical_prices.pkl',
                 metadata_file='sp500_metadata.pkl'):
        """
        Initialize the data manager.
        
        Parameters:
        -----------
        components_url : str
            URL to the historical components CSV
        data_file : str
            Path to the main data pickle file
        metadata_file : str
            Path to the metadata pickle file
        """
        self.components_url = components_url
        self.data_file = data_file
        self.metadata_file = metadata_file
        
    def load_historical_components(self):
        """
        Load historical S&P 500 components from CSV file.
        """
        print("Loading historical S&P 500 components...")
        df = pd.read_csv(self.components_url)
        print(f"Loaded data with shape: {df.shape}")
        return df
    
    def get_all_unique_tickers(self, components_df):
        """
        Extract all unique tickers that have ever been in the S&P 500.
        """
        all_tickers = set()
        
        for row in components_df["tickers"]:
            tickers = row.split(",")
            all_tickers.update(tickers)
        
        all_tickers = sorted([str(ticker).strip() for ticker in all_tickers if pd.notna(ticker)])
        
        print(f"Found {len(all_tickers)} unique tickers")
        return all_tickers
    
    def create_membership_map(self, components_df):
        """
        Create a mapping of dates to tickers that were in the S&P 500 on that date.
        
        Returns:
        --------
        membership_map : dict
            Dictionary with date strings as keys and sets of tickers as values
        """
        print("\nCreating S&P 500 membership map...")
        
        membership_map = {}
        
        # The components_df has dates as column names (except 'date' column if it exists)

        for idx, row in components_df.iterrows():
            tickers = row["tickers"].split(",")
            date = row["date"] # should be a string
            membership_map[date] = tickers 
        
        print(f"Created membership map with {len(membership_map)} dates")
        return membership_map
    
    def add_sp500_membership_flag(self, price_df, components_df):
        """
        Add a boolean column 'in_sp500' to the price dataframe indicating whether
        the ticker was in the S&P 500 on that date.
        
        Parameters:
        -----------
        price_df : pd.DataFrame
            DataFrame with price data
        components_df : pd.DataFrame
            DataFrame with historical S&P 500 components
            
        Returns:
        --------
        price_df : pd.DataFrame
            DataFrame with added 'in_sp500' column
        """
        print("\nAdding S&P 500 membership flag...")
        
        # Create membership map
        membership_map = self.create_membership_map(components_df)
        
        # Sort membership dates for efficient lookup
        sorted_dates = sorted(membership_map.keys())
        
        # Function to check if ticker was in S&P 500 on a given date
        def is_in_sp500(row):
            ticker = row['Ticker']
            date_str = row['Date'].strftime('%Y-%m-%d')
            
            # Find the most recent membership date that is <= the row date
            # This assumes membership stays the same until the next recorded change
            applicable_date = None
            for mem_date in sorted_dates:
                if mem_date <= date_str:
                    applicable_date = mem_date
                else:
                    break
            
            if applicable_date is None:
                return False
            
            return ticker in membership_map[applicable_date]
        
        # Add the flag
        print("Processing membership flags (this may take a few minutes)...")
        price_df['in_sp500'] = price_df.apply(is_in_sp500, axis=1)
        
        # Print statistics
        total_rows = len(price_df)
        in_sp500_rows = price_df['in_sp500'].sum()
        print(f"Membership flag added:")
        print(f"  Total rows: {total_rows:,}")
        print(f"  Rows where ticker was in S&P 500: {in_sp500_rows:,} ({100*in_sp500_rows/total_rows:.1f}%)")
        print(f"  Rows where ticker was NOT in S&P 500: {total_rows - in_sp500_rows:,} ({100*(total_rows - in_sp500_rows)/total_rows:.1f}%)")
        
        return price_df
    
    def download_stock_data(self, ticker, start_date='1960-01-01', end_date=None):
        """
        Download historical stock data for a single ticker.
        Goes back as far as possible (default: 1960).
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                return None
            
            df['Ticker'] = ticker
            df.reset_index(inplace=True)
            
            # Ensure Date is datetime
            df['Date'] = pd.to_datetime(df['Date'])
            
            return df
        
        except Exception as e:
            print(f"Error downloading {ticker}: {str(e)}")
            return None
    
    def download_all_stocks(self, tickers, start_date='1970-01-01', end_date=None, delay=0.1):
        """
        Download historical data for all tickers.
        Goes back as far as possible (default: 1970).
        """
        all_data = []
        failed_tickers = []
        
        print(f"\nDownloading data for {len(tickers)} tickers...")
        print(f"Date range: {start_date} to {end_date or 'today'}")
        
        for ticker in tqdm(tickers, desc="Downloading"):
            df = self.download_stock_data(ticker, start_date, end_date)
            
            if df is not None and not df.empty:
                all_data.append(df)
            else:
                failed_tickers.append(ticker)
            
            time.sleep(delay)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"\nSuccessfully downloaded data for {len(all_data)} tickers")
            print(f"Failed to download: {len(failed_tickers)} tickers")
            
            return combined_df, failed_tickers
        else:
            print("No data was downloaded successfully!")
            return None, failed_tickers
    
    def save_data(self, df, metadata=None):
        """
        Save DataFrame and metadata to pickle files.
        """
        # Save main data
        df.to_pickle(self.data_file)
        print(f"\nData saved to {self.data_file}")
        print(f"File size: {os.path.getsize(self.data_file) / (1024*1024):.2f} MB")
        
        # Create and save metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_rows': len(df),
            'unique_tickers': df['Ticker'].nunique(),
            'date_range': (df['Date'].min().strftime('%Y-%m-%d'), 
                          df['Date'].max().strftime('%Y-%m-%d')),
            'columns': df.columns.tolist(),
            'tickers': sorted(df['Ticker'].unique().tolist())
        })
        
        pd.to_pickle(metadata, self.metadata_file)
        
        self._print_summary(metadata)
        
        return metadata
    
    def _print_summary(self, metadata):
        """
        Print data summary.
        """
        print(f"\nData Summary:")
        print(f"  Last update: {metadata['last_update']}")
        print(f"  Total rows: {metadata['total_rows']:,}")
        print(f"  Unique tickers: {metadata['unique_tickers']}")
        print(f"  Date range: {metadata['date_range'][0]} to {metadata['date_range'][1]}")
        print(f"  Columns: {metadata['columns']}")
    
    def load_data(self):
        """
        Load existing data from pickle file.
        
        Returns:
        --------
        df : pd.DataFrame
            The loaded data
        metadata : dict
            Metadata about the data
        """
        if not os.path.exists(self.data_file):
            print(f"Data file {self.data_file} not found!")
            return None, None
        
        print(f"Loading data from {self.data_file}...")
        df = pd.read_pickle(self.data_file)
        
        metadata = None
        if os.path.exists(self.metadata_file):
            metadata = pd.read_pickle(self.metadata_file)
            self._print_summary(metadata)
        else:
            print("Metadata file not found, creating new metadata...")
            metadata = {
                'total_rows': len(df),
                'unique_tickers': df['Ticker'].nunique(),
                'date_range': (df['Date'].min().strftime('%Y-%m-%d'), 
                              df['Date'].max().strftime('%Y-%m-%d')),
                'columns': df.columns.tolist()
            }
        
        return df, metadata
    
    def update_data(self, full_refresh=False, delay=0.1):
        """
        Update existing data with new data.
        
        Parameters:
        -----------
        full_refresh : bool
            If True, re-download all historical data (to capture Yahoo corrections)
            If False, only download data from the last date in existing data
        delay : float
            Delay between API calls
        
        Returns:
        --------
        df : pd.DataFrame
            Updated dataframe
        """
        print("\n" + "="*60)
        print("UPDATING S&P 500 DATA")
        print("="*60)
        
        # Load existing data
        existing_df, metadata = self.load_data()
        
        if existing_df is None:
            print("No existing data found. Performing initial download...")
            return self.initial_download(delay=delay)
        
        # Get all tickers (including new ones)
        components_df = self.load_historical_components()
        all_tickers = self.get_all_unique_tickers(components_df)
        
        # Determine which tickers to update
        existing_tickers = set(existing_df['Ticker'].unique())
        new_tickers = set(all_tickers) - existing_tickers
        
        if new_tickers:
            print(f"\nFound {len(new_tickers)} new tickers: {sorted(list(new_tickers))[:10]}...")
        
        if full_refresh:
            print("\nFull refresh mode: Re-downloading all historical data...")
            print("This will overwrite existing data with any corrections from Yahoo Finance.")
            
            # Re-download everything from as far back as possible
            start_date = '1970-01-01'
            tickers_to_update = all_tickers
            
        else:
            print("\nIncremental update mode: Downloading only new data...")
            
            # Get the latest date in existing data
            max_date = existing_df['Date'].max()
            start_date = (max_date + timedelta(days=1)).strftime('%Y-%m-%d')
            
            print(f"Last date in existing data: {max_date.strftime('%Y-%m-%d')}")
            print(f"Downloading from: {start_date}")
            
            # Update existing tickers + download new tickers from beginning
            tickers_to_update = all_tickers
        
        # Download new/updated data
        new_data, failed = self.download_all_stocks(
            tickers_to_update, 
            start_date=start_date,
            delay=delay
        )
        
        if new_data is None or new_data.empty:
            print("\nNo new data downloaded.")
            return existing_df
        
        if full_refresh:
            # In full refresh mode, replace all data
            print("\nReplacing old data with refreshed data...")
            combined_df = new_data
            
            # Add membership flag
            combined_df = self.add_sp500_membership_flag(combined_df, components_df)
        else:
            # In incremental mode, merge new data with existing
            print("\nMerging new data with existing data...")
            
            # For new tickers, download full history
            if new_tickers:
                print(f"Downloading full history for {len(new_tickers)} new tickers...")
                new_ticker_data, new_failed = self.download_all_stocks(
                    list(new_tickers),
                    start_date='1970-01-01',  # Go back as far as possible
                    delay=delay
                )
                
                if new_ticker_data is not None:
                    new_data = pd.concat([new_data, new_ticker_data], ignore_index=True)
                    failed.extend(new_failed)
            
            # Add membership flag to new data
            new_data = self.add_sp500_membership_flag(new_data, components_df)
            
            # Remove 'in_sp500' column from existing data if it exists (we'll recalculate)
            if 'in_sp500' in existing_df.columns:
                existing_df = existing_df.drop(columns=['in_sp500'])
            
            # Combine with existing data
            # Remove duplicates (in case of overlapping dates), keeping the newer data
            combined_df = pd.concat([existing_df, new_data], ignore_index=True)
            combined_df = combined_df.sort_values(['Ticker', 'Date'])
            
            # Remove duplicates, keeping the last occurrence (newer data)
            combined_df = combined_df.drop_duplicates(
                subset=['Ticker', 'Date'], 
                keep='last'
            )
            
            # Recalculate membership flag for all data to ensure consistency
            print("\nRecalculating S&P 500 membership for all data...")
            combined_df = self.add_sp500_membership_flag(combined_df, components_df)
        
        # Save updated data
        print("\nSaving updated data...")
        metadata = self.save_data(combined_df, metadata)
        
        # Save failed tickers
        if failed:
            pd.DataFrame({'ticker': failed}).to_csv('failed_tickers.csv', index=False)
            print(f"\nFailed tickers saved to failed_tickers.csv")
        
        print("\n" + "="*60)
        print("UPDATE COMPLETE")
        print("="*60)
        
        return combined_df
    
    def initial_download(self, start_date='1970-01-01', delay=0.1):
        """
        Perform initial download of all data.
        
        Parameters:
        -----------
        start_date : str
            Start date for historical data. Default is 1970-01-01 to go back as far as possible.
        delay : float
            Delay between API calls in seconds
        """
        print("\n" + "="*60)
        print("INITIAL DOWNLOAD OF S&P 500 DATA")
        print(f"Going back to: {start_date}")
        print("="*60)
        
        # Load components
        components_df = self.load_historical_components()
        all_tickers = self.get_all_unique_tickers(components_df)
        
        # Save ticker list
        with open('all_tickers.txt', 'w') as f:
            f.write('\n'.join(all_tickers))
        print(f"Ticker list saved to all_tickers.txt")
        
        # Download all data
        price_data, failed = self.download_all_stocks(
            all_tickers, 
            start_date=start_date,
            delay=delay
        )
        
        if price_data is not None:
            # Add S&P 500 membership flag
            price_data = self.add_sp500_membership_flag(price_data, components_df)
            
            # Save data
            self.save_data(price_data)
            
            # Save failed tickers
            if failed:
                pd.DataFrame({'ticker': failed}).to_csv('failed_tickers.csv', index=False)
                print(f"\nFailed tickers saved to failed_tickers.csv")
            
            print("\n" + "="*60)
            print("INITIAL DOWNLOAD COMPLETE")
            print("="*60)
            
            return price_data
        else:
            print("No data downloaded!")
            return None


def main():
    """
    Main execution function with options.
    """
    import sys
    
    # Create manager
    manager = SP500DataManager()
    
    # Default start date (go back as far as possible)
    start_date = '1970-01-01'
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        # Check for optional start date argument
        if '--start-date' in sys.argv:
            idx = sys.argv.index('--start-date')
            if idx + 1 < len(sys.argv):
                start_date = sys.argv[idx + 1]
                print(f"Using custom start date: {start_date}")
        
        if command == 'load':
            # Just load and display info
            df, metadata = manager.load_data()
            if df is not None:
                print(f"\nFirst few rows:")
                print(df.head())
        
        elif command == 'update':
            # Incremental update
            df = manager.update_data(full_refresh=False)
        
        elif command == 'refresh':
            # Full refresh (re-download everything)
            df = manager.update_data(full_refresh=True)
        
        elif command == 'download':
            # Initial download with optional custom start date
            df = manager.initial_download(start_date=start_date)
        
        else:
            print(f"Unknown command: {command}")
            print("Available commands: download, load, update, refresh")
            print("\nOptional arguments:")
            print("  --start-date YYYY-MM-DD    Set custom start date (only for 'download' command)")
            print("\nExamples:")
            print("  python download_sp500_data.py download")
            print("  python download_sp500_data.py download --start-date 1980-01-01")
    
    else:
        # Default behavior: check if data exists, if not download, if yes update
        if os.path.exists(manager.data_file):
            print("Existing data found. Use 'update' to add new data or 'refresh' to re-download all.")
            df, metadata = manager.load_data()
        else:
            print("No existing data found. Starting initial download...")
            print(f"Going back to: {start_date}")
            df = manager.initial_download(start_date=start_date)


if __name__ == "__main__":
    main()
