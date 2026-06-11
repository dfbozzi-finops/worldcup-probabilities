import pandas as pd
import requests

def test_fbref():
    url = "https://fbref.com/en/comps/1/2022/stats/2022-World-Cup-Stats"
    try:
        tables = pd.read_html(
            url, 
            storage_options={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        print(f"Found {len(tables)} tables")
        print(f"Found {len(tables)} tables")
        if tables:
            df = tables[0]
            # Print columns to understand structure
            print("Columns:", df.columns.tolist()[:10])
            
            # Drop multi-index columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(0)
                
            # Filter for Messi
            messi = df[df['Player'].str.contains('Messi', na=False, case=False)]
            print("Messi Stats:")
            print(messi[['Player', 'Squad', 'Gls', 'Ast', 'PK', 'PKatt']].to_dict('records'))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_fbref()
