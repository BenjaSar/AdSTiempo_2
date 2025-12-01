import subprocess
import sys
import time

DEBUG = False

# Configuration settings
class APIConfig:
    TICKER = 'BTC-USD'
    START_DATE = '2020-01-01'
    END_DATE = '2020-02-01'
    MAX_RETRIES = 2
    MIN_DATA_POINTS = 10 # A check to ensure a meaningful amount of data is returned

try:
    import yfinance as yf 
except ImportError:
    yf = None # Set to None if the module is missing

def install_or_reinstall_yfinance():
    """Helper function to perform a clean install."""
    if DEBUG: print("\n👩🏼‍⚕️ Attempting clean install/reinstall of yfinance...")
    try:
        # Uninstall first
        subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'yfinance'], 
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Install
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'yfinance'], 
                       check=True, stdout=subprocess.DEVNULL)
        print("✅ yfinance successfully installed/reinstalled.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during pip operation: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during installation: {e}")
        return False


def attempt_data_download(ticker, start, end):
    """Tries to import yfinance and download data."""
    # Check for module availability (handled at module level now)
    if yf is None:
        print("⚠️ yfinance module is missing.")
        return None
    
    # Attempt data download
    try:        
        if DEBUG: print(f"📥 Attempting download for {ticker}...")
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        
        if df.empty:
            if DEBUG: print("⚠️ Data returned empty.")
            return None
        
        return df
    
    except Exception as e:
        print(f"⚠️ API or download error: {e}")
        return None
    
if __name__ == '__main__':
    # Main Logic with Retries
    data_df = None
    DEBUG = True

    print(f"\n🏥 Starting API verification and repair process…")
    for attempt in range(1, APIConfig.MAX_RETRIES + 1):
        print(f"\n--- Attempt {attempt} of {APIConfig.MAX_RETRIES} ---")
        data_df = attempt_data_download(APIConfig.TICKER, APIConfig.START_DATE, APIConfig.END_DATE)
        
        if data_df is not None and len(data_df) >= APIConfig.MIN_DATA_POINTS:
            print("✅ Data download successful and meets minimum data points requirement.")
            break
        else:
            print("⚠️ Data download failed or insufficient data. Initiating repair...")
            success = install_or_reinstall_yfinance()
            if not success:
                print("❌ Repair failed. Aborting further attempts.")
                break
            time.sleep(2) # Brief pause before retrying
    else:
        print("🛑 All attempts exhausted without successful data retrieval.")

    # --- Final Output ---
    print("\n" + "="*40)
    if data_df is not None:
        print("DATA SAMPLE (5 rows):")
        print(data_df.head())
    else:
        print("🛑 Failed to retrieve data after all attempts.")