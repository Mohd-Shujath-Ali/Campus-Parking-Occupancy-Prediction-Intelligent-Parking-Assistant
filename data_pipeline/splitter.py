import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

class ChronologicalSplitter:
    """
    Splits continuous zone-level time-series data chronologically into Train (70%),
    Validation (15%), and Holdout Test (15%) partitions with an enforced safety buffer 
    to eliminate cross-boundary lookahead leakage.
    """
    def __init__(
        self,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        buffer_hours: int = 1
    ):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.buffer_hours = buffer_hours

    def split_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        unique_timestamps = np.sort(df['timestamp'].unique())
        total_steps = len(unique_timestamps)
        
        train_end_idx = int(total_steps * self.train_ratio)
        val_end_idx = int(total_steps * (self.train_ratio + self.val_ratio))
        
        train_max_time = unique_timestamps[train_end_idx]
        val_start_time = train_max_time + pd.Timedelta(hours=self.buffer_hours)
        val_max_time = unique_timestamps[val_end_idx]
        test_start_time = val_max_time + pd.Timedelta(hours=self.buffer_hours)

        df_train = df[df['timestamp'] <= train_max_time].copy()
        df_val = df[(df['timestamp'] >= val_start_time) & (df['timestamp'] <= val_max_time)].copy()
        df_test = df[df['timestamp'] >= test_start_time].copy()

        return df_train, df_val, df_test
