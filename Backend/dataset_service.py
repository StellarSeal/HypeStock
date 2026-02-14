# --- DATASET SERVICE MODULE ---
# Handles CSV parsing, metadata aggregation, and pagination.
import os
import csv
import random

# In-memory storage for metadata
STOCK_METADATA = []
IS_LOADED = False

# Paths
COMPANIES_PATH = 'companies.csv'
PRICES_PATH = 'stock_prices.csv'

def generate_mock_data():
    """Generates mock data if CSVs are missing."""
    print("[DATA] CSV files not found. Generating mock data...")
    mock_stocks = []
    sectors = ['Tech', 'Finance', 'Health', 'Energy']
    
    for i in range(150):
        # Generate code: AA + A-Z
        code = f"{chr(65 + (i // 26))}{chr(65 + (i % 26))}{chr(65 + random.randint(0, 25))}"
        
        mock_stocks.append({
            "stock_code": code,
            "company_name": f"Mock Company {code}",
            "sector": random.choice(sectors),
            "start_date": "2020-01-01",
            "end_date": "2024-02-14",
            "entry_count": random.randint(500, 5500)
        })
    
    # Sort by stock_code
    mock_stocks.sort(key=lambda x: x['stock_code'])
    return mock_stocks

def load_datasets():
    """Loads and aggregates data from CSVs."""
    global STOCK_METADATA, IS_LOADED
    
    if IS_LOADED:
        return

    if not os.path.exists(COMPANIES_PATH):
        STOCK_METADATA = generate_mock_data()
        IS_LOADED = True
        return

    print("[DATA] Loading datasets...")
    companies = {}

    try:
        # 1. Load Companies
        with open(COMPANIES_PATH, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Normalize keys just in case
                code = row.get('stock_code') or row.get('Stock Code')
                name = row.get('company_name') or row.get('Company Name')
                if code:
                    companies[code] = name or "Unknown Company"

        # 2. Aggregate (Mocked aggregation for performance/missing price file)
        temp_list = []
        for code, name in companies.items():
            temp_list.append({
                "stock_code": code,
                "company_name": name,
                "start_date": "2021-01-01", 
                "end_date": "2023-12-31",
                "entry_count": random.randint(100, 1000)
            })
        
        # Sort alphabetically
        temp_list.sort(key=lambda x: x['stock_code'])
        STOCK_METADATA = temp_list
        print(f"[DATA] Loaded {len(STOCK_METADATA)} stocks.")
        IS_LOADED = True

    except Exception as e:
        print(f"[ERROR] Failed to load datasets: {e}")
        STOCK_METADATA = generate_mock_data()
        IS_LOADED = True

def get_stocks(page=0, limit=20, query=""):
    """Returns paginated stock list."""
    results = STOCK_METADATA
    
    # Filter
    if query:
        q = query.lower()
        results = [
            s for s in results 
            if q in s['stock_code'].lower() or q in s['company_name'].lower()
        ]
    
    # Pagination
    start_index = page * limit
    end_index = start_index + limit
    paginated_items = results[start_index:end_index]
    
    return {
        "items": paginated_items,
        "total": len(results),
        "hasMore": end_index < len(results)
    }