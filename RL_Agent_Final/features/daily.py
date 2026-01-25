import pandas as pd
import numpy as np

class DailyFeatures:
    def __init__(self, csv_path='Data/Processed/Indicators.csv'):
        self.csv_path = csv_path
        self.df = None
        
    def load(self):
        try:
            self.df = pd.read_csv(self.csv_path)
            # Ensure Date parsing
            # The file has 'Date' column
            if 'Date' in self.df.columns:
                self.df['Date'] = pd.to_datetime(self.df['Date'], utc=True).dt.tz_convert(None).dt.date
            else:
                 print("Daily Indicators: 'Date' column not found.")
                 return False
                 
            self.df = self.df.set_index('Date')
            self.df = self.df.sort_index()
            
            # Filter strictly numeric columns once
            self.df = self.df.select_dtypes(include=[np.number])
            self.feat_dim = len(self.df.columns)
            # print(f"Daily Indicators loaded. Nuemric features: {self.feat_dim}")
            
            return True
        except Exception as e:
            print(f"Error loading Daily Indicators: {e}")
            return False
            
    def get_for_day(self, timestamp):
        """
        Get daily features for a specific timestamp (converted to date).
        Returns values as array.
        """
        if self.df is None:
            return np.zeros(1) # Fallback safe
            
        target_date = timestamp.date()
        
        try:
            if target_date in self.df.index:
                row = self.df.loc[target_date]
                
                # Check if we got duplicates (DataFrame)
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0] # Take first
                    
                return row.values
            else:
                return np.zeros(self.feat_dim) # Consistent Size
        except Exception as e:
            # print(f"Daily lookup error: {e}")
            return np.zeros(self.feat_dim)
