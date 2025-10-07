"""
Data Lake operations for time series data
Handles file-based storage in Parquet format
"""
import os
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path


def write_sgs_parquet(code: str, observations: List[Dict[str, Any]], base_dir: str = "data/raw") -> int:
    """
    Write time series observations to Parquet file in data lake
    
    Args:
        code: Series identifier code
        observations: List of dicts with 'date' and 'value' keys
        base_dir: Base directory for data lake storage
    
    Returns:
        Number of rows written
    """
    if not observations:
        return 0
    
    try:
        # Create DataFrame from observations
        df = pd.DataFrame(observations)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        
        # Create directory structure
        output_dir = Path(base_dir) / "source=SGS" / f"code={code}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write to Parquet file
        output_file = output_dir / f"data_{code}.parquet"
        df.to_parquet(output_file, index=False, engine="pyarrow")
        
        print(f"Wrote {len(df)} rows to {output_file}")
        return len(df)
    
    except Exception as e:
        print(f"Error writing parquet for series {code}: {e}")
        return 0


def read_sgs_parquet(code: str, base_dir: str = "data/raw") -> pd.DataFrame:
    """
    Read time series observations from Parquet file
    
    Args:
        code: Series identifier code
        base_dir: Base directory for data lake storage
    
    Returns:
        DataFrame with time series data
    """
    try:
        input_file = Path(base_dir) / "source=SGS" / f"code={code}" / f"data_{code}.parquet"
        
        if not input_file.exists():
            print(f"Parquet file not found: {input_file}")
            return pd.DataFrame()
        
        df = pd.read_parquet(input_file, engine="pyarrow")
        return df
    
    except Exception as e:
        print(f"Error reading parquet for series {code}: {e}")
        return pd.DataFrame()
