import pytest # Import pytest for the framework
import time
import subprocess
from unittest.mock import patch, MagicMock

# Import the functions and classes to be tested
from utils.fix_api import APIConfig, install_or_reinstall_yfinance, attempt_data_download

print(APIConfig)
# # --- Configuration ---
# APIConfig.TICKER = 'BTC-USD'
# APIConfig.START_DATE = '2020-01-01'
# APIConfig.END_DATE = '2020-02-01'
# APIConfig.MAX_RETRIES = 2
# APIConfig.MIN_DATA_POINTS = 10 # A check to ensure a meaningful amount of data is returned

# TESTS

def test_yfinance_availability_and_data_access():
    """
    Test function that includes retry and repair logic.
    """
    data_df = None
    attempt = 0

    while data_df is None and attempt < APIConfig.MAX_RETRIES:
        print(f"\n--- Test Attempt {attempt + 1} of {APIConfig.MAX_RETRIES} ---")
        
        # 1. Try to download data
        data_df = attempt_data_download(APIConfig.TICKER, APIConfig.START_DATE, APIConfig.END_DATE)

        # 2. If download failed
        if data_df is None:
            # If it's the first attempt, or any subsequent failure, try to fix it
            print("Triggering repair sequence...")
            # NOTE: We are relying on the real install/reinstall here, which is slow but validates the whole flow.
            repair_successful = install_or_reinstall_yfinance()
            
            if not repair_successful and attempt == APIConfig.MAX_RETRIES - 1:
                # If repair failed on the last attempt, raise a definite error
                pytest.fail("Failed to install/reinstall yfinance after all attempts.")
            
            # Since the repair function was run, we must reset data_df 
            # and let the loop run again for the next attempt.
            data_df = None 

        attempt += 1
        if data_df is None and attempt < APIConfig.MAX_RETRIES:
            print("Waiting 3 seconds before next attempt...")
            time.sleep(3)

    # 3. Final Assertions (The actual test outcome)
    assert data_df is not None, "Failed to retrieve financial data after all attempts."
    
    # Assert that the DataFrame is not trivially small
    assert len(data_df) > APIConfig.MIN_DATA_POINTS, f"Data size ({len(data_df)}) is too small."
    
    print(f"\n✅ Test Passed! Retrieved {len(data_df)} data points.")


## Testing `attempt_data_download` Isolation

@patch('utils.fix_api.yf', new=None)
def test_data_download_import_error():
    """Test that download fails gracefully when yfinance is None (module is missing)."""
    # Now this test simply checks the 'if yf is None' block in attempt_data_download.
    result = attempt_data_download(APIConfig.TICKER, APIConfig.START_DATE, APIConfig.END_DATE)
    assert result is None, "Did not return None when yfinance module is missing."


def test_data_download_empty_dataframe():
    """Test that download fails when the API returns an an empty DataFrame."""
    # FIX: Create a mock download object that returns a DataFrame with empty=True
    mock_df = MagicMock(empty=True)
    
    # FIX: Patch the specific yf.download method used inside attempt_data_download
    with patch('utils.fix_api.yf.download', return_value=mock_df) as mock_download:
        result = attempt_data_download(APIConfig.TICKER, APIConfig.START_DATE, APIConfig.END_DATE)
        
        # We must assert that the function correctly processes the empty check and returns None
        assert result is None, "Did not return None for an empty DataFrame result."
        mock_download.assert_called_once()


def test_data_download_api_exception():
    """Test that download fails gracefully on a generic API/network exception."""
    # FIX: Patch the specific yf.download method to raise an exception
    with patch('utils.fix_api.yf.download', side_effect=Exception("Mock API Error")) as mock_download:
        result = attempt_data_download(APIConfig.TICKER, APIConfig.START_DATE, APIConfig.END_DATE)
        
        # We must assert that the function correctly catches the exception and returns None
        assert result is None, "Did not return None for an API exception."
        mock_download.assert_called_once()

def test_data_download_success():
    """Test that download succeeds when valid data is returned."""
    # FIX: Create a mock DataFrame that is NOT empty
    mock_df = MagicMock(empty=False, __len__=MagicMock(return_value=100))
    
    # FIX: Patch the specific yf.download method to return the mock data
    with patch('utils.fix_api.yf.download', return_value=mock_df) as mock_download:
        result = attempt_data_download(APIConfig.TICKER, APIConfig.START_DATE, APIConfig.END_DATE)
        
        # We must assert that the function returns the mock data
        assert result is mock_df, "Did not return the mock DataFrame on success."
        mock_download.assert_called_once()


## Testing `install_or_reinstall_yfinance` Isolation

# Patch Target: utils.fix_api.subprocess.run
@patch('utils.fix_api.subprocess.run')
def test_install_failure_called_process_error(mock_run):
    """Test that install fails when pip returns a non-zero exit code."""
    mock_run.side_effect = [
        MagicMock(), # Successful uninstall mock (First call)
        subprocess.CalledProcessError(returncode=1, cmd='pip install yfinance') # Failed install mock (Second call)
    ]
    
    result = install_or_reinstall_yfinance()
    
    assert mock_run.call_count == 2, "Did not attempt both uninstall and install."
    assert result is False, "Did not return False on CalledProcessError."


# Patch Target: utils.fix_api.subprocess.run
@patch('utils.fix_api.subprocess.run', side_effect=Exception("Unexpected System Error"))
def test_install_failure_unexpected_exception(mock_run):
    """Test that install fails on an unexpected exception (e.g., OS issue)."""
    
    result = install_or_reinstall_yfinance()
    
    mock_run.assert_called() 
    assert result is False, "Did not return False on unexpected Exception."