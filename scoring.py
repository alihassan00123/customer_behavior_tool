import pandas as pd

def calculate_rfm(df):
    # ✅ Case 1: Data is already prepared with RFM features
    if all(col in df.columns for col in ['Recency', 'TransactionCount', 'TotalSpent', 'CustomerID']):
        # Rename columns to match standard RFM
        df['Frequency'] = df['TransactionCount']
        df['Monetary'] = df['TotalSpent']
        
        # Select and format RFM columns
        rfm = df[['CustomerID', 'Recency', 'Frequency', 'Monetary']].copy()
        rfm.set_index('CustomerID', inplace=True)
        return rfm

    # ✅ Case 2: Raw transaction data is provided
    elif all(col in df.columns for col in ['InvoiceDate', 'CustomerID', 'InvoiceNo', 'Quantity', 'UnitPrice']):
        # Convert invoice date to datetime
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Calculate total amount per row
        df['TotalSum'] = df['Quantity'] * df['UnitPrice']
        
        # Define snapshot date (usually 1 day after the last transaction)
        snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

        # Group data by CustomerID and calculate RFM
        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
            'InvoiceNo': 'nunique',                                   # Frequency
            'TotalSum': 'sum'                                         # Monetary
        })
        
        # Rename columns
        rfm.columns = ['Recency', 'Frequency', 'Monetary']
        return rfm

    # ❌ Case 3: Missing required columns
    else:
        raise ValueError("❌ Data must either contain:\n"
                         "- Raw: InvoiceDate, CustomerID, InvoiceNo, Quantity, UnitPrice\n"
                         "- Or Ready: Recency, TransactionCount, TotalSpent, CustomerID")
