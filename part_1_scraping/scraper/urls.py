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
    def __init__(self, base_url: str, csv_directory: str = 'part_1_scraping/data') -> None:
        # Base URL for searching properties (both houses and apartments)
        self.base_url = base_url
        # Directory to save CSV file
        self.csv_directory = csv_directory
        # Custom User-Agent to mimic browser request
        self.headers = {"User-Agent": 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:130.0) Gecko/20100101 Firefox/130.0'}
        # Timeout for HTTP requests
        self.timeout = httpx.Timeout(50.0)
        # List to store URLs for all properties (houses and apartments)
        self.property_urls = []

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
            # We only need one URL to handle both houses and apartments
            tasks.append(self._scrape_property_type(session))
            
            # Run all scraping tasks concurrently
            await asyncio.gather(*tasks)

    async def _scrape_property_type(self, session):
        """
        Scrape property links for houses and apartments together 
        by iterating through search result pages
        """
        page_num = 1
        while True:
            # Construct URL for each search results page
            url = f"{self.base_url}?countries=BE&page={page_num}&orderBy=relevance"
            print(f"Fetching page {page_num}...")

            try:
                # Send HTTP GET request
                response = await session.get(url, headers=self.headers, timeout=self.timeout)
                
                if response.status_code != 200:
                    print(f"Error fetching page {page_num}. Status code: {response.status_code}")
                    break
                
                # Extract property links from the search results page
                links_found = self._parse_search_page_for_property_links(response.text)
                
                if not links_found:
                    print(f"No listings found on page {page_num}. Ending scrape.")
                    break
                
                page_num += 1
                
            except Exception as e:
                print(f"Error occurred while fetching page {page_num}: {e}")
                break

    def _parse_search_page_for_property_links(self, html: str) -> bool:
        """
        Parse HTML of search results page and extract unique property URLs
        
        Args:
            html (str): HTML content of the search results page
        
        Returns:
            bool: Whether any new links were found
        """
        soup = BeautifulSoup(html, 'html.parser')
        links_found = False
        
        # Find all links to individual property listings using regex
        for link in soup.find_all('a', href=re.compile(r'^https://www\.immoweb\.be/en/classified/(apartment|house)/for-sale/')):
            url = link['href']
            # Add link only if it's not already in the list
            if url not in self.property_urls:
                self.property_urls.append(url)
                links_found = True
        
        return links_found

    def save_urls_to_csv(self) -> None:
        """
        Save collected property URLs to a single CSV file, 
        removing any duplicates
        """
        # Remove duplicates
        self.property_urls = list(set(self.property_urls))

        # Create a DataFrame for all properties
        urls_df = pd.DataFrame(self.property_urls, columns=['url'])

        # Save URLs to CSV file
        urls_df.to_csv(os.path.join(self.csv_directory, 'urls_properties.csv'), index=False)

        # Print the number of links saved
        print(f"Property links saved: {len(self.property_urls)}")

def main():
    # Define the base URL for searching both apartments and houses
    base_url = "https://www.immoweb.be/en/search/house-and-apartment/for-sale"
    
    # Initialize and run the URL scraper
    get_urls = GetURLs(base_url)
    get_urls.run_as(get_urls.fetch_and_parse())
    get_urls.save_urls_to_csv()

if __name__ == "__main__":
    main()
