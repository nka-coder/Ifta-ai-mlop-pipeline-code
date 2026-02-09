import json
import boto3
import pandas as pd
import numpy as np
from datetime import datetime
import openpyxl
from io import BytesIO
import logging
import re
import urllib.parse

CANADIAN_LOCATIONS = {
    'AB': ['Edmonton', 'Calgary', 'Red Deer', 'Lethbridge', 'Medicine Hat', 'Fort McMurray'],
    'BC': ['Vancouver', 'Kelowna', 'Langley', 'Burnaby', 'Surrey', 'Victoria', 'Prince George'],
    'MB': ['Winnipeg', 'Brandon', 'Thompson', 'Portage la Prairie'],
    'ON': ['Toronto', 'Ottawa', 'Barrie', 'Kingston', 'Sudbury', 'Thunder Bay', 'Willowdale', 'Caledon'],
    'SK': ['Regina', 'Saskatoon', 'Prince Albert', 'Moose Jaw', 'Swift Current'],
    'QC': ['Montreal', 'Quebec City', 'Laval', 'Gatineau'],
    'NL': ['St. Johns', 'Mount Pearl'],
    'NB': ['Fredericton', 'Saint John', 'Moncton'],
    'NS': ['Halifax', 'Sydney', 'Truro'],
    'PE': ['Charlottetown', 'Summerside']
}

# Clients
s3_client = boto3.client('s3')
bedrock = boto3.client('bedrock-runtime')
comprehend = boto3.client('comprehend')
sagemaker = boto3.client('sagemaker-runtime')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client('s3')
bedrock = boto3.client('bedrock-runtime')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    try:
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'])
        destination_bucket = 'NAME OF DESTINATION BUCKET'

        # Extract vehicle ID 
        vehicle_id = "abcd" # Hardcoded for now. It will be extracted from the filename
        
        records = process_driver_logs_with_ai(bucket, key, vehicle_id)
        
        if not records:
            return {'statusCode': 200, 'body': 'No records found'}

        save_to_parquet_with_metadata(records, destination_bucket, key)
        
        return {'statusCode': 200, 'records_processed': len(records)}
    except Exception as e:
        logger.error(f"Lambda failed: {str(e)}")
        raise e

def process_driver_logs_with_ai(bucket, key, vehicle_id):
    response = s3_client.get_object(Bucket=bucket, Key=key)
    df = pd.read_excel(BytesIO(response['Body'].read()), engine='openpyxl')
    
    # 1. Rename columns with 'raw_' prefix
    df = df.rename(columns={col: f"raw_{col}" for col in df.columns})
    
    # 2. FIX: Convert NaT/NaN to JSON-serializable formats
    # Convert all datetime columns to string
    for col in df.select_dtypes(include=['datetime', 'datetimetz']).columns:
        df[col] = df[col].apply(lambda x: x.strftime('%Y-%m-%d') if pd.notnull(x) else None)
    
    # Replace NaN/NaT with None (JSON null)
    df = df.replace({np.nan: None})
    raw_records = df.to_dict(orient='records')
    
    # 3. AI Enhancement
    enhanced_data = enhance_with_bedrock(raw_records)
    
    final_output = []
    for record in enhanced_data:
        # Map renamed raw columns back to logic
        origin_val = record.get('raw_Origin') or record.get('origin', 'Unknown')
        dest_val = record.get('raw_Destination') or record.get('destination', 'Unknown')
        dist_val = record.get('raw_Distance_km') or 0
        
        geo_origin = geocode_with_ai(str(origin_val))
        geo_dest = geocode_with_ai(str(dest_val))
        
        jurisdiction = estimate_jurisdiction_with_ai(geo_origin, geo_dest, dist_val)
        
        final_output.append({
            **record,
            'vehicle_id': vehicle_id,
            'origin_city': geo_origin['city'],
            'origin_jurisdiction': geo_origin['province'],
            'destination_city': geo_dest['city'],
            'destination_jurisdiction': geo_dest['province'],
            'Distance_km': float(dist_val),
            'start_odometer': float(record.get('raw_Odometer_Start', 0) or 0),
            'end_odometer': float(record.get('raw_Odometer_End', 0) or 0),
            'jurisdiction_breakdown': jurisdiction,
            'processed_at': datetime.utcnow().strftime('%Y-%m-%d') 
        })
        
    return final_output

def enhance_with_bedrock(data):
    """Claude 3 Messages API implementation"""
    prompt = f"""
    Act as an IFTA specialist. Here is a list of driver log rows: {json.dumps(data)}
    1. Fix date formats to YYYY-MM-DD.
    2. Ensure every row has a numeric Distance_km.
    3. Return ONLY a valid JSON list of objects. No conversational text.
    """
    
    try:
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        })

        response = bedrock.invoke_model(
            modelId='anthropic.claude-3-sonnet-20240229-v1:0',
            body=body
        )
        
        res_bytes = json.loads(response['body'].read())
        text_content = res_bytes['content'][0]['text']
        
        # Regex to extract JSON list in case AI adds chatter
        match = re.search(r'\[.*\]', text_content, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(text_content)
    except Exception as e:
        logger.error(f"Bedrock failed, returning raw data: {e}")
        return data

def geocode_with_ai(location_text):
    """Smarter geocoding using local dictionary and regex"""
    location_text = str(location_text).strip()
    
    # 1. Check for Province codes already in the text (e.g., "Barrie, ON")
    prov_map = {'AB': 'AB', 'BC': 'BC', 'ON': 'ON', 'MB': 'MB', 'QC': 'QC', 'SK': 'SK', 'NS': 'NS', 'NB': 'NB'}
    for code in prov_map:
        if re.search(rf'\b{code}\b', location_text, re.IGNORECASE):
            city = location_text.split(',')[0].strip()
            return {'city': city, 'province': code}

    # 2. Check our CANADIAN_LOCATIONS dictionary for the city name
    # This catches "Brandon" and knows it is "MB"
    for province, cities in CANADIAN_LOCATIONS.items():
        for city in cities:
            if city.lower() in location_text.lower():
                return {'city': city, 'province': province}

    # 3. Fallback for typos (like "Lankey") - Let Bedrock handle it later or return Unknown
    return {'city': location_text, 'province': 'Unknown'}

def estimate_jurisdiction_with_ai(origin, destination, distance_km):
    """Calculates the split, using Bedrock only if provinces are different"""
    
    # If same province, 100% split is easy
    if origin['province'] == destination['province'] and origin['province'] != 'Unknown':
        return {origin['province']: 100.0}
    
    # If one is Unknown or they are different, ask Bedrock to be the expert
    try:
        prompt = f"""
        Given a trip from {origin['city']}, {origin['province']} to {destination['city']}, {destination['province']}.
        Total distance: {distance_km} km.
        Provide the IFTA jurisdiction breakdown (provinces traveled through).
        Return ONLY valid JSON like: {{"ON": 80.0, "QC": 20.0}}
        """
        
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 200,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0
        })

        response = bedrock.invoke_model(modelId='anthropic.claude-3-haiku-20240307-v1:0', body=body)
        res_body = json.loads(response['body'].read())
        result_text = res_body['content'][0]['text']
        
        # Extract JSON from AI response
        match = re.search(r'\{.*\}', result_text)
        return json.loads(match.group()) if match else {origin['province']: 50.0, destination['province']: 50.0}

    except Exception as e:
        logger.warning(f"Jurisdiction AI failed: {e}")
        return {origin['province']: 50.0, destination['province']: 50.0}


def save_to_parquet_with_metadata(records, bucket, key):
    """Saves the final records to Parquet"""
    if not records: return
    
    df = pd.DataFrame(records)
    parquet_buffer = BytesIO()
    df.to_parquet(parquet_buffer, index=False, engine='pyarrow')
    
    new_key = key.rsplit('.', 1)[0] + '.parquet'
    s3_client.put_object(
        Bucket=bucket,
        Key=new_key,
        Body=parquet_buffer.getvalue()
    )
    logger.info(f"Successfully saved to {new_key}")
