import pytest # Import pytest for the framework
import time
from fix_api import install_or_reinstall_yfinance, attempt_data_download

# --- Configuration ---
TICKER = 'BTC-USD'
START_DATE = '2020-01-01'
END_DATE = '2020-02-01'
MAX_RETRIES = 2
MIN_DATA_POINTS = 10 # A check to ensure a meaningful amount of data is returned

def test_yfinance_availability_and_data_access():
    """
    Test function that includes retry and repair logic.
    """
    data_df = None
    attempt = 0

    while data_df is None and attempt < MAX_RETRIES:
        print(f"\n--- Test Attempt {attempt + 1} of {MAX_RETRIES} ---")
        
        # 1. Try to download data
        data_df = attempt_data_download(TICKER, START_DATE, END_DATE)

        # 2. If download failed
        if data_df is None:
            # If it's the first attempt, or any subsequent failure, try to fix it
            print("Triggering repair sequence...")
            repair_successful = install_or_reinstall_yfinance()
            
            if not repair_successful and attempt == MAX_RETRIES - 1:
                # If repair failed on the last attempt, raise a definite error
                pytest.fail("Failed to install/reinstall yfinance after all attempts.")
            
            # Since the repair function was run, we must reset data_df 
            # and let the loop run again for the next attempt.
            data_df = None 

        attempt += 1
        if data_df is None and attempt < MAX_RETRIES:
            print("Waiting 3 seconds before next attempt...")
            time.sleep(3)

    # 3. Final Assertions (The actual test outcome)
    assert data_df is not None, "Failed to retrieve financial data after all attempts."
    
    # Assert that the DataFrame is not trivially small
    assert len(data_df) > MIN_DATA_POINTS, f"Data size ({len(data_df)}) is too small."
    
    print(f"\n✅ Test Passed! Retrieved {len(data_df)} data points.")