#!/usr/bin/env python3
"""
TED Data Retriever Module

This module handles retrieving data from the TED API based on specified criteria.
It provides functionality to fetch procurement notices and convert them to a usable format.

Author: Your Name
Date: May 16, 2025
"""

import os
import json
import requests
import pandas as pd
import csv
import time
from datetime import datetime

class TEDDataRetriever:
    """Class for retrieving data from the TED API"""
    
    def __init__(self, data_dir="data"):
        self.headers = {'Content-type': 'application/json', 'Accept': 'text/plain', "Charset": "UTF-8"}
        self.base_url = "https://tedweb.api.ted.europa.eu/v3/notices/search"
        self.data_dir = data_dir
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
    
    def get_notices_page(self, page_number, start_date, end_date, max_bid_amount=None, country=None, limit=100):
        """
        Get a single page of notices from the TED API and return the results.
        Args:
            page_number (int): Page number to retrieve
            start_date (str): Start date in format YYYYMMDD
            end_date (str): End date in format YYYYMMDD
            max_bid_amount (float, optional): Maximum bid amount to filter by
            country (str, optional): Country code to filter by (e.g., 'GRC')
            limit (int): Number of results per page
        Returns:
            list: List of notice data dictionaries
        """
        # Use the exact same query format as in test.py
        query = f"publication-date>={start_date}<={end_date}"
        
        # Add country filter if specified
        if country:
            query += f" AND organisation-country-buyer={country}"
        
        # Prepare request parameters
        params = {
            "query": query,
            "fields": [
                "notice-identifier",
                "estimated-value-cur-lot",
                "no-negocaition-necessary-lot",
                "direct-award-justification-text-proc",
                "legal-basis",
                "procedure-identifier",
                "winner-size",
                "winner-selection-status",
                "notice-type",
                "estimated-value-cur-proc",
                "total-value",
                "framework-value-notice",
                "subcontracting-percentage",
                "subcontracting-value",
                "direct-award-justification-proc",
                "ipi-measures-applicable-lot",
                "procedure-accelerated",
                "legal-basis-proc",
                "legal-basis-text",
                "eu-registration-number",
                "exclusion-grounds",
                "framework-buyer-categories-lot",
                "dps-usage-lot",
                "accessibility-lot",
                "winner-owner-nationality",
                "organisation-country-buyer",
            ],
            "page": page_number,
            "limit": limit
        }
        
        # Prepare request parameters
        params = {
            "query": query,
            "fields": [
                "notice-identifier",
                "estimated-value-cur-lot",
                "no-negocaition-necessary-lot",
                "direct-award-justification-text-proc",
                "legal-basis",
                "procedure-identifier",
                "winner-size",
                "winner-selection-status",
                "notice-type",
                "estimated-value-cur-proc",
                "total-value",
                "framework-value-notice",
                "subcontracting-percentage",
                "subcontracting-value",
                "direct-award-justification-proc",
                "ipi-measures-applicable-lot",
                "procedure-accelerated",
                "legal-basis-proc",
                "legal-basis-text",
                "eu-registration-number",
                "exclusion-grounds",
                "framework-buyer-categories-lot",
                "dps-usage-lot",
                "accessibility-lot",
                "winner-owner-nationality",
                "organisation-country-buyer",
            ],
            "page": page_number,
            "limit": limit
        }
        
        print(f"Making API request for page {page_number}...")
        try:
            response = requests.post(self.base_url, json=params, headers=self.headers)
            if response.status_code == 200:
                data = json.loads(response.text)
                notices = data.get("notices", [])
                print(f"Successfully retrieved {len(notices)} notices from page {page_number}")
                
                # Filter by max_bid_amount if specified
                if max_bid_amount is not None and notices:
                    filtered_notices = []
                    for notice in notices:
                        total_value = notice.get("total-value", 0)
                        if total_value is None or float(total_value or 0) <= float(max_bid_amount):
                            filtered_notices.append(notice)
                    print(f"Filtered to {len(filtered_notices)} notices within budget {max_bid_amount}")
                    return filtered_notices
                return notices
            else:
                print(f"Error on page {page_number}: {response.status_code}")
                print(response.text)
                return []
        except Exception as e:
            print(f"Exception during API request: {str(e)}")
            return []

    def flatten_notice(self, notice):
        """
        Flatten a nested notice structure into a single-level dictionary.
        Args:
            notice (dict): Notice data dictionary
        Returns:
            dict: Flattened notice dictionary
        """
        flat_notice = {}
        for key, value in notice.items():
            if isinstance(value, dict):
                # Flatten nested dictionaries with dot notation
                for nested_key, nested_value in value.items():
                    flat_notice[f"{key}.{nested_key}"] = nested_value
            elif isinstance(value, list):
                # Join list values with a separator
                flat_notice[key] = "|".join(str(item) for item in value)
            else:
                flat_notice[key] = value
        return flat_notice
    
    def save_to_csv(self, df, timestamp=None):
        """
        Save DataFrame to CSV with timestamp
        Args:
            df (pd.DataFrame): DataFrame to save
            timestamp (str, optional): Timestamp to use in filename, defaults to current time
        Returns:
            str: Path to saved file
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_file = os.path.join(self.data_dir, f"ted_notices_{timestamp}.csv")
        df.to_csv(output_file, index=False)
        print(f"Raw data saved to {output_file}")
        
        return output_file
    
    def fetch_notices(self, start_date, end_date, max_bid_amount=None, country=None, max_pages=5):
        """
        Fetch notices from the TED API based on specified criteria
        Args:
            start_date (str): Start date in format YYYYMMDD
            end_date (str): End date in format YYYYMMDD
            max_bid_amount (float, optional): Maximum bid amount
            country (str, optional): Country code
            max_pages (int): Maximum number of pages to fetch
        Returns:
            tuple: (pd.DataFrame, str) - DataFrame containing notices and path to saved CSV
        """
        all_notices = []
        
        # Process pages one by one
        for page_num in range(1, max_pages + 1):
            notices = self.get_notices_page(
                page_num, 
                start_date, 
                end_date, 
                max_bid_amount, 
                country
            )
            
            if not notices:
                print(f"No notices found on page {page_num}, stopping pagination")
                break
                
            # Flatten notices and add to the list
            flattened_notices = [self.flatten_notice(notice) for notice in notices]
            all_notices.extend(flattened_notices)
            
            # Add a delay between requests to avoid rate limiting
            time.sleep(1)
        
        if not all_notices:
            print("No notices were found with the specified criteria")
            return pd.DataFrame(), None
            
        # Convert to DataFrame
        df = pd.DataFrame(all_notices)
        print(f"Successfully fetched {len(df)} notices")
        
        # Save raw data to CSV
        output_file = self.save_to_csv(df)
        
        return df, output_file


# If run directly, perform a test fetch
if __name__ == "__main__":
    # Example usage
    retriever = TEDDataRetriever()
    
    # Fetch notices for the last month
    today = datetime.now()
    end_date = today.strftime("%Y%m%d")
    start_date = (today.replace(day=1) - pd.DateOffset(months=1)).strftime("%Y%m%d")
    
    print(f"Fetching notices from {start_date} to {end_date}")
    df, output_file = retriever.fetch_notices(
        start_date=start_date,
        end_date=end_date,
        max_pages=2
    )
    
    if not df.empty:
        print(f"Retrieved {len(df)} notices")
        print(f"Sample columns: {', '.join(df.columns[:5])}")
        print("\nSample data:")
        print(df.head(2))
    else:
        print("No data retrieved")