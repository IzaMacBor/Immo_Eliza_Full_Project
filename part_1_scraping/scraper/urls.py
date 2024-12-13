import os
import pandas as pd
import asyncio
import httpx
from bs4 import BeautifulSoup
import threading
import re

# Thread wrapper for running asynchronous coroutines in a thread-safe manner
class RunThread(threading.Thread):
    def __init__(self, coro):
        self.coro = coro
        self.result = None
        super().__init__()

    def run(self) -> None:
        # Run the coroutine using asyncio
        self.result = asyncio.run(self.coro)

class GetURLs:
    def __init__(self, base_urls: list, csv_directory: str = 'part_1_scraping/data/raw_csv') -> None:
        # List of base search URLs for different property types
        self.base_urls = base_urls
        # Directory to save CSV files
        self.csv_directory = csv_directory
        # Custom User-Agent to mimic browser request
        self.headers = {"User-Agent": 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:130.0) Gecko/20100101 Firefox/130.0'}
        # Timeout for HTTP requests
        self.timeout = httpx.Timeout(50.0)
        # Dictionary to store URLs for different property types
        self.property_urls = {
            'apartments': [],
            'houses': []
        }

        # Create CSV directory if it doesn't exist
        if not os.path.exists(self.csv_directory):
            os.makedirs(self.csv_directory)

    def run_as(self, coro) -> None:
        """
        Run an asynchronous coroutine in a thread-safe manner, 
        handling both running and non-running event loops
        """
        try:
            global_loop = asyncio.get_running_loop()
        except RuntimeError:
            global_loop = None
        
        if global_loop and global_loop.is_running():
            # If an event loop is already running, use a separate thread
            thread = RunThread(coro)
            thread.start()
            thread.join()
            return thread.result
        else:
            # If no event loop is running, create a new one
            print('Starting new event loop')
            try:
                return asyncio.run(coro)
            except Exception as e:
                print(f"Error during scraping: {e}")
                return None

    async def fetch_and_parse(self) -> None:
        """
        Fetch and parse URLs for houses and apartments simultaneously 
        using asynchronous HTTP requests
        """
        async with httpx.AsyncClient() as session:
            tasks = []
            for base_url in self.base_urls:
                # Determine property type based on URL
                property_type = 'apartments' if 'apartment' in base_url else 'houses'
                tasks.append(self._scrape_property_type(session, base_url, property_type))
            
            # Run all scraping tasks concurrently
            await asyncio.gather(*tasks)

    async def _scrape_property_type(self, session, base_url, property_type):
        """
        Scrape property links for a specific property type 
        by iterating through search result pages
        """
        page_num = 1
        while True:
            # Construct URL for each search results page
            url = f"{base_url}?countries=BE&page={page_num}&orderBy=relevance"
            print(f"Fetching {property_type} page {page_num}...")
            
            try:
                # Send HTTP GET request
                response = await session.get(url, headers=self.headers, timeout=self.timeout)
                
                if response.status_code != 200:
                    print(f"Error fetching {property_type} page {page_num}. Status code: {response.status_code}")
                    break
                
                # Extract property links from the search results page
                links_found = self._parse_search_page_for_property_links(response.text, property_type)
                
                if not links_found:
                    print(f"No {property_type} listings found on page {page_num}. Ending scrape.")
                    break
                
                page_num += 1
                
            except Exception as e:
                print(f"Error occurred while fetching {property_type} page {page_num}: {e}")
                break

    def _parse_search_page_for_property_links(self, html: str, property_type: str) -> bool:
        """
        Parse HTML of search results page and extract unique property URLs
        
        Args:
            html (str): HTML content of the search results page
            property_type (str): Type of property being scraped
        
        Returns:
            bool: Whether any new links were found
        """
        soup = BeautifulSoup(html, 'html.parser')
        links_found = False
        
        # Find all links to individual property listings using regex
        for link in soup.find_all('a', href=re.compile(r'^https://www\.immoweb\.be/en/classified/(apartment|house)/for-sale/')):
            url = link['href']
            # Add link only if it's not already in the list
            if url not in self.property_urls[property_type]:
                self.property_urls[property_type].append(url)
                links_found = True
        
        return links_found

    def save_urls_to_csv(self) -> None:
        """
        Save collected property URLs to CSV files, 
        removing any duplicates
        """
        # Remove duplicates
        for property_type in self.property_urls:
            self.property_urls[property_type] = list(set(self.property_urls[property_type]))

        # Create DataFrames for apartments and houses
        apartments_df = pd.DataFrame(self.property_urls['apartments'], columns=['url'])
        houses_df = pd.DataFrame(self.property_urls['houses'], columns=['url'])

        # Save URLs to CSV files
        apartments_df.to_csv(os.path.join(self.csv_directory, 'urls_apartments.csv'), index=False)
        houses_df.to_csv(os.path.join(self.csv_directory, 'urls_houses.csv'), index=False)

        # Print the number of links saved
        print(f"Apartments links saved: {len(self.property_urls['apartments'])}")
        print(f"Houses links saved: {len(self.property_urls['houses'])}")

def main():
    # Define base URLs for searching apartments and houses
    base_urls = [
        "https://www.immoweb.be/en/search/apartment/for-sale",
        "https://www.immoweb.be/en/search/house/for-sale"
    ]
    # Initialize and run the URL scraper
    get_urls = GetURLs(base_urls)
    get_urls.run_as(get_urls.fetch_and_parse())
    get_urls.save_urls_to_csv()

if __name__ == "__main__":
    main()