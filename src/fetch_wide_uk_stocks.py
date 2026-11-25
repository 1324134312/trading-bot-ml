"""
Fetch Wide Range of UK Stocks - 100+ stocks
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yfinance as yf
import pandas as pd
import time
from tqdm import tqdm

# FTSE 100 + FTSE 250 stocks (100+ UK stocks)
WIDE_UK_STOCKS = [
    # FTSE 100 (already have some)
    "BP.L", "SHEL.L", "GSK.L", "AZN.L", "ULVR.L", "DGE.L", "RIO.L", 
    "BHP.L", "VOD.L", "BT-A.L", "BARC.L", "LLOY.L", "NXT.L", "BAE.L",
    "AAL.L", "PRU.L", "ANTO.L", "AHT.L", "REL.L", "SSE.L",
    
    # Add more FTSE 100
    "TSCO.L", "LSEG.L", "CRDA.L", "CTEC.L", "HLMA.L", "DSM.L",
    "HIK.L", "SN.L", "SGE.L", "RS1.L", "KGF.L", "INF.L", "ITRK.L",
    "BYG.L", "TATE.L", "WTB.L", "RTO.L", "CWK.L", "ABDN.L", "CCH.L",
    
    # Add FTSE 250 (smaller cap)
    "TRIT.L", "DOM.L", "BOY.L", "RKN.L", "CTY.L", "BBY.L", "DARK.L",
    "SCPT.L", "GNT.L", "FDM.L", "FUND.L", "BAB.L", "HMSO.L", "MCG.L",
    "PETS.L", "AGL.L", "TM17.L", "ALF.L", "CCL.L", "CW.L",
    
    # UK REITs (different sector)
    "SBRY.L", "SSPG.L", "IGG.L", "SHI.L", "LAND.L", "TED.L", "SHB.L",
    
    # More diverse sectors
    "FINS.L", "GAW.L", "VSVS.L", "UAI.L", "MKS.L", "JD.L", "SAFE.L",
    "ASC.L", "GLEN.L", "BLND.L", "FERG.L", "RDSA.L", "RMS.L", "TED.L",
]

def fetch_wide_uk_stocks():
    """Fetch 5-minute data for 100+ UK stocks"""
    
    print("=" * 70)
    print("🇬🇧 Fetching WIDE Range of UK Stocks")
    print(f"   Total stocks: {len(WIDE_UK_STOCKS)}")
    print("   Interval: 5-minute bars")
    print("   Period: 7 days (Yahoo limit)")
    print("=" * 70)
    
    os.makedirs("data/uk/raw_wide", exist_ok=True)
    
    results = []
    successful = 0
    
    for ticker in tqdm(WIDE_UK_STOCKS, desc="Fetching"):
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="7d", interval="5m")
            
            if not data.empty and len(data) > 50:
                data.reset_index(inplace=True)
                filename = f"data/uk/raw_wide/{ticker}_5m_7d.csv"
                data.to_csv(filename, index=False)
                
                results.append({
                    'ticker': ticker,
                    'bars': len(data)
                })
                successful += 1
                
        except Exception as e:
            # Fail silently
            pass
        
        time.sleep(0.5)
    
    # Summary
    print(f"\n{'=' * 70}")
    print("✅ Summary")
    print(f"{'=' * 70}")
    print(f"Successfully fetched: {successful}/{len(WIDE_UK_STOCKS)} stocks")
    
    if results:
        total_bars = sum(r['bars'] for r in results)
        print(f"Total 5-minute bars: {total_bars:,}")
        print(f"Average per stock: {total_bars / successful:.0f}")
        
        print(f"\n📊 This gives us:")
        print(f"   ~{total_bars:,} total data points")
        print(f"   ~{total_bars // 1000}k training samples")
        print(f"   Much better than {successful * 100} bars from daily data!")
        
    return results

if __name__ == "__main__":
    results = fetch_wide_uk_stocks()
    
    print(f"\n🎯 Next: Process features and train model!")
    print(f"   This will give us a robust UK day trading model!")

