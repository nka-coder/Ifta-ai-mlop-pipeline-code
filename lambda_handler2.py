import json
import boto3
import io
import re
import zipfile
import pandas as pd
from datetime import datetime
import urllib.parse

# AWS Clients
textract = boto3.client('textract')
s3 = boto3.client('s3')

# Standard IFTA Queries
IFTA_QUERIES = [
    {"Text": "What is the total price?", "Alias": "TOTAL_PRICE"},
    {"Text": "What is the fuel type (Diesel or Gasoline)?", "Alias": "FUEL_TYPE"},
    {"Text": "How many liters or gallons were purchased?", "Alias": "VOLUME"},
    {"Text": "What is the date of purchase?", "Alias": "DATE"},
    {"Text": "What is the station address or province?", "Alias": "LOCATION"}
]

PROVINCE_MAP = {
    'ALBERTA': 'AB', 'ONTARIO': 'ON', 'MANITOBA': 'MB', 'SASKATCHEWAN': 'SK',
    'BRITISH COLUMBIA': 'BC', 'QUEBEC': 'QC', 'NEW BRUNSWICK': 'NB',
    'NOVA SCOTIA': 'NS', 'PRINCE EDWARD ISLAND': 'PE', 'NEWFOUNDLAND': 'NL'
}

def lambda_handler(event, context):
    try:
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'])
        destination_bucket = 'NAME OF DESTINATION BUCKET'
        
        # Extract vehicle ID
        vehicle_id = "abcd" # Hardcoded for now. It will be extracted from the filename
        
        response = s3.get_object(Bucket=bucket, Key=key)
        file_content = response['Body'].read()
        
        raw_image_data = []
        with zipfile.ZipFile(io.BytesIO(file_content)) as z:
            image_files = [f for f in z.namelist() if f.startswith('word/media/')]
            for img_path in sorted(image_files):
                raw_image_data.append(z.read(img_path))

        extracted_list = []
        for idx, img_bytes in enumerate(raw_image_data):
            res = textract.analyze_document(Document={'Bytes': img_bytes}, FeatureTypes=['QUERIES'], QueriesConfig={'Queries': IFTA_QUERIES})
            data = parse_textract_results(res)
            
            extracted_list.append({
                "raw_date": data.get("DATE"), 
                "raw_fuel_type": data.get("FUEL_TYPE"),
                "raw_volume": data.get("VOLUME"), 
                "raw_price": data.get("TOTAL_PRICE"),
                "raw_location": data.get("LOCATION"), 
                "receipt_index": idx + 1
            })

        final_df = clean_and_format_data(extracted_list, vehicle_id)

        if not final_df.empty:
            output_key = key.rsplit('.', 1)[0] + "_extracted.parquet"
            parquet_buffer = io.BytesIO()
            final_df.to_parquet(parquet_buffer, index=False, engine='pyarrow')
            s3.put_object(Bucket=destination_bucket, Key=output_key, Body=parquet_buffer.getvalue())
            return {"status": "success", "vehicle_id": vehicle_id}

    except Exception as e:
        print(f"Error: {str(e)}")
        raise e

def parse_textract_results(response):
    results = {}
    block_map = {b['Id']: b for b in response['Blocks']}
    for b in response['Blocks']:
        if b['BlockType'] == 'QUERY':
            alias = b['Query']['Alias']
            results[alias] = None
            if 'Relationships' in b:
                for rel in b['Relationships']:
                    if rel['Type'] == 'ANSWER':
                        for ans_id in rel['Ids']:
                            ans_block = block_map.get(ans_id)
                            if ans_block:
                                results[alias] = ans_block.get('Text')
    return results

def split_location_robust(val):
    if not val: return None, None
    val_upper = str(val).upper()
    found_prov = None
    for name, code in PROVINCE_MAP.items():
        if name in val_upper:
            found_prov = code
            break
    if not found_prov:
        prov_codes = list(PROVINCE_MAP.values())
        match_code = re.search(r'\b(' + '|'.join(prov_codes) + r')\b', val_upper)
        if match_code:
            found_prov = match_code.group(1)

    clean_val = re.sub(r'\d+', '', val_upper)
    clean_val = re.sub(r'[A-Z]\d[A-Z]\s?\d[A-Z]\d', '', clean_val)
    parts = [p.strip() for p in re.split(r'[,\s-]+', clean_val) if p.strip()]
    
    city = parts[-1].capitalize() if parts else None
    if found_prov and len(parts) > 1:
        if parts[-1] in PROVINCE_MAP or parts[-1] in PROVINCE_MAP.values():
            city = parts[-2].capitalize()
    return city, found_prov

def clean_and_format_data(raw_records, vehicle_id):
    if not raw_records: return pd.DataFrame()
    df = pd.DataFrame(raw_records)
    
    # 1. Driver ID for Glue Join
    df['vehicle_id'] = vehicle_id
    
    # 2. Robust Date Parsing
    df['PurchaseDate'] = pd.to_datetime(df['raw_date'], errors='coerce').dt.strftime('%Y-%m-%d')
    
    # If raw_date was unparseable or missing, PurchaseDate will remain None.

    def parse_numeric(val):
        if not val: return 0.0
        match = re.search(r"(\d+\.?\d*)", str(val))
        return float(match.group(1)) if match else 0.0

    # 3. Numeric Formatting
    df['Quantity'] = df['raw_volume'].apply(parse_numeric)
    df['TotalAmount'] = df['raw_price'].apply(parse_numeric)
    df['FuelType'] = df['raw_fuel_type'].str.upper().fillna("UNKNOWN")

    # 4. Jurisdiction & City
    df[['StationCity', 'Jurisdiction']] = df['raw_location'].apply(lambda x: pd.Series(split_location_robust(x)))
    
    # 5. Final preparation for Parquet
    df = df.where(pd.notnull(df), None)
    
    return df

