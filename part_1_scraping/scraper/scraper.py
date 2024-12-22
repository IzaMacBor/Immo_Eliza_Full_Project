import os
import csv
import json
import asyncio
import httpx
import pandas as pd
from bs4 import BeautifulSoup
import logging

class PropertyScraper:
    def __init__(self, 
                 urls_directory=r'C:\Users\izama\Documents\GitHub\Immo_Eliza_Full_Project\part_1_scraping\data', 
                 output_directory=r'C:\Users\izama\Documents\GitHub\Immo_Eliza_Full_Project\part_1_scraping\data', 
                 max_concurrent_requests=10,
                 start_index=None,  # Default is None to scrape all URLs
                 end_index=None):   # Default is None to scrape all URLs
        """
        Initialize the PropertyScraper with configuration parameters.
        
        :param urls_directory: Directory containing URL CSV files
        :param output_directory: Directory to save processed property data
        :param max_concurrent_requests: Maximum number of concurrent HTTP requests
        :param start_index: Starting index for URL range to scrape (optional)
        :param end_index: Ending index for URL range to scrape (optional)
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)

        # Directories and paths
        self.urls_directory = urls_directory
        self.output_directory = output_directory
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_directory, exist_ok=True)

        # HTTP request configuration
        self.headers = {
            "User-Agent": 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:130.0) Gecko/20100101 Firefox/130.0'
        }
        self.timeout = httpx.Timeout(50.0)
        
        # Concurrency control
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)
        
        # Data storage
        self.raw_data_dict = {
            'properties': {}
        }
        self.problem_links = {
            'properties': []
        }
        
        # URL index range
        self.start_index = start_index
        self.end_index = end_index

    def load_property_urls(self):
        """
        Load property URLs from a single CSV file containing all property URLs.
        
        :return: List of property URLs
        """
        urls = []
        
        try:
            urls_df = pd.read_csv(os.path.join(self.urls_directory, 'urls_properties.csv'))
            
            # If no start_index or end_index is provided, load all URLs
            if self.start_index is None or self.end_index is None:
                urls = urls_df['url'].tolist()
                self.logger.info(f"Loaded all property URLs ({len(urls)})")
            else:
                # Select the URLs in the range [start_index, end_index]
                urls = urls_df['url'].iloc[self.start_index:self.end_index].tolist()
                self.logger.info(f"Loaded {len(urls)} property URLs from index {self.start_index} to {self.end_index}")
            
            return urls
        
        except FileNotFoundError as e:
            self.logger.error(f"URL CSV file not found: {e}")
            return urls

    async def fetch_property_data(self, url, session):
        """
        Fetch and extract raw data for a single property.
        
        :param url: Property listing URL
        :param session: HTTP client session
        :return: Extracted property data or None
        """
        try:
            async with self.semaphore:
                response = await session.get(url, headers=self.headers, timeout=self.timeout)
                
                if response.status_code != 200:
                    self.logger.warning(f"Failed to fetch {url}. Status code: {response.status_code}")
                    self.problem_links['properties'].append(url)
                    return None

                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract JSON data from script tag
                script_tag = soup.find_all('script', attrs={'type':'text/javascript'})[1]
                window_classifier = str(script_tag).replace(
                    '<script type="text/javascript">', ''
                ).replace('</script>', '').strip().replace('window.classified = ', '').strip(';')
                
                try:
                    raw_data = json.loads(window_classifier)
                    
                    # Remove unnecessary data
                    for key in ['customers', 'premiumProjectPage', 'media']:
                        raw_data.pop(key, None)
                    
                    return raw_data
                except json.JSONDecodeError as e:
                    self.logger.error(f"JSON decoding error for {url}: {e}")
                    self.problem_links['properties'].append(url)
                    return None

        except Exception as e:
            self.logger.error(f"Error processing {url}: {e}")
            self.problem_links['properties'].append(url)
            return None

    async def collect_property_data(self):
        """
        Collect raw data for properties.
        """
        urls = self.load_property_urls()
        
        async with httpx.AsyncClient() as session:
            tasks = [
                self.fetch_property_data(url, session) 
                for url in urls
            ]
            results = await asyncio.gather(*tasks)
            
            # Store results, filtering out None values
            self.raw_data_dict['properties'] = {
                url: data 
                for url, data in zip(urls, results) 
                if data is not None
            }

    def clean_property_data(self):
        """
        Clean and process raw property data.
        
        :return: List of cleaned property dictionaries
        """
        cleaned_data = []
        
        
        for link, raw_data in self.raw_data_dict['properties'].items():
            # Skip life annuity sales
            if raw_data.get("flags", {}).get("isLifeAnnuitySale", False):
                continue
            
            # Safely retrieve data from the raw_data dictionary
            property_info = {
                'link': link,
                'property_id': raw_data.get('id', 'null'),
                'locality_name': raw_data.get('property', {}).get('location', {}).get('locality', 'null'),
                'postal_code': raw_data.get('property', {}).get('location', {}).get('postalCode', 'null'),
                'latitude': raw_data.get('property', {}).get('location', {}).get('latitude', 'null'),
                'longitude': raw_data.get('property', {}).get('location', {}).get('longitude', 'null'),
                'price': raw_data.get('price', {}).get('mainValue', 'null'),
                'type_of_property': raw_data.get('property', {}).get('type', 'null'),
                'subtype_of_property': raw_data.get('property', {}).get('subtype', 'null'),
                'construction_year': (
                    raw_data.get('property', {}).get('building', {}).get('constructionYear', 'null') 
                    if raw_data and raw_data.get('property') and raw_data.get('property').get('building') 
                    else 'null'
                ),
                'living_area': raw_data.get('property', {}).get('netHabitableSurface', 'null'),
                'bedrooms': (
                    raw_data.get('property', {}).get('bedroomCount', 'null') 
                    if raw_data and raw_data.get('property') and isinstance(raw_data.get('property'), dict) 
                else 'null'
                ),
                'nr_bathrooms': raw_data.get('property', {}).get('bathroomCount', 'null'),
                'furnished': self._parse_boolean(raw_data.get('transaction', {}).get('sale', {}).get('isFurnished', None)),
                'open_fire': 1 if raw_data.get('property', {}).get('fireplaceExists', False) else 0,
                'terrace_surface': (
                    raw_data.get('property', {}).get('terraceSurface', 'null') 
                    if raw_data.get('property', {}).get('hasTerrace', False) 
                    else 'null'
                ),
                'garden': (
                    raw_data.get('property', {}).get('gardenSurface', 'null') 
                    if raw_data.get('property', {}).get('hasGarden', False) 
                    else 'null'
                ),
                'facades': (
                    raw_data.get('property', {}).get('building', {}).get('facadeCount', 0)  
                    if raw_data and raw_data.get('property') and raw_data.get('property').get('building') 
                    else 0
                ),
                'swimming_pool': 1 if raw_data.get('property', {}).get('hasSwimmingPool', False) else 0,
                'land_area': self._get_safe_value(raw_data.get('property', {}).get('land', {}), 'surface'),
                'equipped_kitchen': (
                    raw_data.get('property', {}).get('kitchen', {}).get('type', 'null') 
                    if raw_data and raw_data.get('property') and raw_data.get('property').get('kitchen') 
                    else 'null'
                ),
                'state_of_building': (
                    raw_data.get('property', {}).get('building', {}).get('condition', 'null')  
                    if raw_data and raw_data.get('property') and isinstance(raw_data.get('property'), dict) 
                    and raw_data.get('property').get('building') and isinstance(raw_data.get('property').get('building'), dict) 
                    else 'null'
                ),
                'type_of_sale': raw_data.get('transaction', {}).get('type', 'null')
            }
            

            cleaned_data.append(property_info)
        
        return cleaned_data    

    def _parse_boolean(self, value):
        """
        Parse boolean values safely.
        
        :param value: Input value to parse
        :return: 1 for True, 0 for False, None otherwise
        """
        if value is True:
            return 1
        elif value is False:
            return 0
        return None

    def _get_safe_value(self, data_dict, key, default_value='null'):
        """
        Safely retrieve a value from a dictionary, returning a default if the key is not found.
        
        :param data_dict: The dictionary to search in
        :param key: The key to look for
        :param default_value: The default value to return if key is not found
        :return: Value of the key or the default value if key doesn't exist
        """
        if data_dict is None:
            return default_value
        return data_dict.get(key, default_value)

    def save_to_csv(self, data):
        """
        Save processed property data to CSV.
        
        :param data: List of property dictionaries
        """
        output_file = os.path.join(self.output_directory, 'raw_properties.csv')
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        
        self.logger.info(f"Saved {len(data)} properties to {output_file}")

    async def run(self):
        """
        Main scraping workflow.
        """
        await self.collect_property_data()
        cleaned_data = self.clean_property_data()
        self.save_to_csv(cleaned_data)
       
        # Log problem links
        if self.problem_links['properties']:
            self.logger.warning(f"Problem links: {len(self.problem_links['properties'])}")

def main():
    """
    Entry point for the script.
    """
    scraper = PropertyScraper(start_index=None, end_index=None)  # Example range
    asyncio.run(scraper.run())

if __name__ == "__main__":
    main()
