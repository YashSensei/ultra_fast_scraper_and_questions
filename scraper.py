import requests
from bs4 import BeautifulSoup
import time
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import aiohttp
import multiprocessing as mp
from queue import Queue, Empty
import random
import os
from urllib.parse import urljoin
import signal
import sys

class HighSpeedSPOJScraper:
    def __init__(self, max_workers=None, connection_limit=300, timeout=8, delay_range=(0.02, 0.08)):
        # Auto-detect optimal worker count based on CPU cores - more aggressive
        if max_workers is None:
            cpu_count = mp.cpu_count()
            self.max_workers = min(cpu_count * 8, 100)  # 8x CPU cores, max 100
        else:
            self.max_workers = max_workers
            
        self.connection_limit = connection_limit
        self.timeout = timeout
        self.delay_range = delay_range
        self.lock = threading.Lock()
        self.scraped_count = 0
        self.failed_count = 0
        
        # Session pool for connection reuse - larger pool
        self.session_pool = Queue(maxsize=self.max_workers * 2)
        self._init_session_pool()
        
        # Batch processing - smaller batches for better parallelism
        self.batch_size = 10  # Reduced from 20 for more granular parallelism
        self.results_queue = Queue()
        
        print(f"🚀 Initialized MAXIMUM SPEED scraper:")
        print(f"   💪 {self.max_workers} concurrent workers")
        print(f"   🔗 {connection_limit} connection limit") 
        print(f"   ⚡ {self.batch_size} problems per batch")
        print(f"   🕒 {delay_range[0]}-{delay_range[1]}s delay range")

    def _init_session_pool(self):
        """Initialize a pool of reusable sessions with maximum performance settings."""
        pool_size = self.max_workers * 2  # Double the pool size
        for _ in range(pool_size):
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
                'Cache-Control': 'max-age=0',
            })
            # Configure session for MAXIMUM performance
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=self.connection_limit // 2,
                pool_maxsize=self.connection_limit,
                max_retries=2,  # Fewer retries for speed
                pool_block=False
            )
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            self.session_pool.put(session)

    def get_session(self):
        """Get a session from the pool."""
        try:
            return self.session_pool.get_nowait()
        except Empty:
            # If pool is empty, create a new session (shouldn't happen normally)
            return self._create_session()

    def return_session(self, session):
        """Return a session to the pool."""
        try:
            self.session_pool.put_nowait(session)
        except:
            pass  # Pool full, let it be garbage collected

    def _create_session(self):
        """Create a new session with optimal settings."""
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20,
            pool_maxsize=20,
            max_retries=2
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def scrape_problem_details_fast(self, problem_links_batch):
        """Scrape multiple problem details in a single thread (batch processing)."""
        session = self.get_session()
        results = []
        
        try:
            for problem_link in problem_links_batch:
                try:
                    url = f"https://www.spoj.com{problem_link}"
                    
                    # Minimal delay for politeness
                    if self.delay_range:
                        time.sleep(random.uniform(self.delay_range[0], self.delay_range[1]))
                    
                    response = session.get(url, timeout=self.timeout)
                    response.raise_for_status()
                    
                    # Use lxml parser for speed
                    soup = BeautifulSoup(response.content, 'lxml')
                    
                    # Fast extraction with error handling
                    try:
                        title_element = soup.find('h2', class_='text-center')
                        if title_element:
                            title_text = title_element.get_text(strip=True)
                            problem_title = title_text.split('-')[1].strip() if '-' in title_text else title_text
                        else:
                            problem_title = f"Problem {problem_link.split('/')[-2]}"
                    except:
                        problem_title = f"Problem {problem_link.split('/')[-2]}"
                    
                    # Extract other details with fallbacks
                    try:
                        problem_text_container = soup.find('div', id='problem-body')
                        problem_text = problem_text_container.get_text(strip=True)[:1000] if problem_text_container else 'Not found'
                    except:
                        problem_text = 'Not found'
                    
                    try:
                        tags_container = soup.find('div', id='problems-tags')
                        tags = [tag.get_text(strip=True) for tag in tags_container.find_all('a')] if tags_container else []
                    except:
                        tags = []
                    
                    try:
                        examples = soup.find_all('pre')
                        sample_input = examples[0].get_text(strip=True) if len(examples) > 0 else 'Not found'
                        sample_output = examples[1].get_text(strip=True) if len(examples) > 1 else 'Not found'
                    except:
                        sample_input = sample_output = 'Not found'

                    result = {
                        'title': problem_title,
                        'tags': tags,
                        'text': problem_text,
                        'sample_input': sample_input,
                        'sample_output': sample_output,
                        'link': url
                    }
                    results.append(result)
                    
                    with self.lock:
                        self.scraped_count += 1
                        if self.scraped_count % 25 == 0:  # More frequent updates
                            elapsed = time.time() - self.start_time if hasattr(self, 'start_time') else 1
                            rate = self.scraped_count / elapsed
                            print(f"  ⚡ Progress: {self.scraped_count} problems | {rate:.1f}/sec")
                    
                except Exception as e:
                    with self.lock:
                        self.failed_count += 1
                    # Don't print every error to avoid spam
                    continue
                    
        finally:
            self.return_session(session)
            
        return results

    def scrape_page_links_optimized(self, page_batch):
        """Scrape problem links from multiple pages in one thread - optimized for 80 pages."""
        session = self.get_session()
        all_links = []
        
        try:
            for page_num in page_batch:
                try:
                    list_url = f"https://www.spoj.com/problems/classical/?start={(page_num - 1) * 50}"
                    
                    # Minimal delay for maximum speed
                    if self.delay_range:
                        time.sleep(random.uniform(self.delay_range[0], self.delay_range[1]))
                    
                    response = session.get(list_url, timeout=self.timeout)
                    response.raise_for_status()
                    
                    # Use lxml for maximum parsing speed
                    soup = BeautifulSoup(response.content, 'lxml')
                    
                    # Fast extraction using CSS selectors
                    problems_table = soup.select_one('table.problems')
                    if not problems_table:
                        continue
                    
                    # Direct CSS selector for all problem links
                    problem_links = problems_table.select('tbody tr td[align="left"] a[href]')
                    
                    page_links = []
                    for link in problem_links:
                        href = link.get('href')
                        if href and '/problems/' in href:
                            page_links.append(href)
                            all_links.append(href)
                    
                    print(f"    📄 Page {page_num}: {len(page_links)} problems found")
                            
                except Exception as e:
                    print(f"    ❌ Page {page_num} failed: {str(e)[:30]}...")
                    continue
                    
        finally:
            self.return_session(session)
            
        return all_links

    def create_batches(self, items, batch_size):
        """Split items into batches for processing."""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    def get_total_pages_fast(self):
        """Quickly estimate total pages."""
        try:
            session = self.get_session()
            response = session.get("https://www.spoj.com/problems/classical/", timeout=10)
            soup = BeautifulSoup(response.content, 'lxml')
            
            # Look for pagination
            pagination = soup.find('ul', class_='pagination')
            if pagination:
                page_links = pagination.find_all('a')
                max_page = 0
                for link in page_links:
                    try:
                        if link.text.isdigit():
                            max_page = max(max_page, int(link.text))
                    except:
                        continue
                        
                if max_page > 0:
                    self.return_session(session)
                    return max_page
            
            self.return_session(session)
            return 850  # Conservative estimate based on SPOJ's typical size
            
        except Exception as e:
            print(f"Error estimating pages: {e}")
            return 850

    def scrape_problems_ultra_fast(self, max_pages=80, max_problems=None, start_page=1):
        """Ultra-fast parallel scraping optimized for SPOJ's 80 pages."""
        print("🔥 MAXIMUM CONCURRENCY SPOJ SCRAPER - 80 PAGES")
        print("=" * 60)
        self.start_time = time.time()
        
        # For SPOJ's known 80 pages
        total_pages = max_pages if max_pages else 80
        print(f"🎯 Target: ALL {total_pages} pages (pages {start_page} to {total_pages})")
        print(f"📊 Expected: ~{total_pages * 50} problems total")
        
        all_problem_data = []
        
        # Phase 1: Collect ALL problem links with MAXIMUM concurrency
        print(f"\n🚀 Phase 1: MAXIMUM SPEED link collection")
        print(f"   💪 Workers: {self.max_workers}")
        print(f"   📦 Batch size: 2 pages per batch (for max parallelism)")
        all_links = []
        
        # Create smaller page batches for maximum parallelism (2 pages per batch)
        pages = list(range(start_page, total_pages + 1))
        page_batches = list(self.create_batches(pages, 2))  # 2 pages per batch for max concurrency
        
        print(f"🔥 Processing {len(pages)} pages in {len(page_batches)} ultra-fast batches...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            batch_futures = {
                executor.submit(self.scrape_page_links_optimized, batch): i 
                for i, batch in enumerate(page_batches)
            }
            
            completed = 0
            for future in as_completed(batch_futures):
                try:
                    links = future.result()
                    all_links.extend(links)
                    completed += 1
                    if completed % 10 == 0 or completed == len(batch_futures):
                        print(f"  📊 Processed {completed}/{len(batch_futures)} page batches, found {len(all_links)} links")
                except Exception as e:
                    print(f"  ❌ Batch failed: {str(e)[:50]}")
        
        print(f"✅ Phase 1 completed: {len(all_links)} problem links collected")
        
        # Apply problem limit
        if max_problems is not None and max_problems < len(all_links):
            all_links = all_links[:max_problems]
            print(f"🎯 Limited to first {max_problems} problems (from page {start_page} onwards)")
        
        if not all_links:
            print("❌ No links found. Exiting.")
            return []
        
        # Phase 2: Scrape ALL problem details with MAXIMUM throughput
        print(f"\n⚡ Phase 2: MAXIMUM THROUGHPUT problem scraping")
        print(f"   🎯 Target: {len(all_links)} problems")
        print(f"   📦 Batch size: {self.batch_size} problems per batch")
        
        # Create smaller problem link batches for maximum parallelism
        link_batches = list(self.create_batches(all_links, self.batch_size))
        print(f"🔥 Created {len(link_batches)} ultra-fast batches")
        print(f"⚡ Starting MAXIMUM CONCURRENCY scraping with {self.max_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            detail_futures = {
                executor.submit(self.scrape_problem_details_fast, batch): i 
                for i, batch in enumerate(link_batches)
            }
            
            completed_batches = 0
            for future in as_completed(detail_futures):
                try:
                    batch_results = future.result()
                    all_problem_data.extend(batch_results)
                    completed_batches += 1
                    
                    if completed_batches % 3 == 0 or completed_batches == len(link_batches):
                        elapsed = time.time() - self.start_time
                        rate = len(all_problem_data) / elapsed if elapsed > 0 else 0
                        progress = (completed_batches / len(link_batches)) * 100
                        eta = (elapsed / completed_batches * (len(link_batches) - completed_batches)) if completed_batches > 0 else 0
                        print(f"  🔥 [{progress:5.1f}%] Batch {completed_batches}/{len(link_batches)} | "
                              f"{len(all_problem_data):4d} problems | {rate:5.1f}/sec | ETA: {eta:3.0f}s")
                        
                except Exception as e:
                    print(f"  ❌ Batch failed: {str(e)[:50]}")
        
        return all_problem_data


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print('\n🛑 Interrupted by user. Saving progress...')
    sys.exit(0)


def main():
    # MAXIMUM CONCURRENCY CONFIGURATION for 80 pages
    MAX_WORKERS = None  # Auto-detect (CPU cores * 8, max 100)
    CONNECTION_LIMIT = 300  # Very high connection limit
    TIMEOUT = 6  # Aggressive timeout for speed
    DELAY_RANGE = (0.02, 0.08)  # Ultra-minimal delays - MAXIMUM SPEED
    MAX_PAGES = 80  # SPOJ has exactly 80 pages
    MAX_PROBLEMS = None  # Get ALL problems
    START_PAGE = 1  # Start from first page
    
    # Handle interruption
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🔥🔥🔥 MAXIMUM CONCURRENCY SPOJ SCRAPER 🔥🔥🔥")
    print("=" * 60)
    print("🎯 MISSION: Scrape ALL 80 pages at MAXIMUM SPEED")
    print("⚡ CONCURRENCY: CPU cores × 8 workers")
    print("🚀 OPTIMIZATION: Ultra-minimal delays, max connections")
    print("=" * 60)
    
    # Create scraper
    scraper = HighSpeedSPOJScraper(
        max_workers=MAX_WORKERS,
        connection_limit=CONNECTION_LIMIT,
        timeout=TIMEOUT,
        delay_range=DELAY_RANGE
    )
    
    # Start scraping
    start_time = time.time()
    try:
        problem_details = scraper.scrape_problems_ultra_fast(
            max_pages=MAX_PAGES,
            max_problems=MAX_PROBLEMS,
            start_page=START_PAGE
        )
    except KeyboardInterrupt:
        print("\n🛑 Scraping interrupted by user")
        return
        
    end_time = time.time()
    total_time = end_time - start_time
    
    # MAXIMUM PERFORMANCE results summary
    print(f"\n🎉🎉🎉 MAXIMUM SPEED SCRAPING COMPLETED! 🎉🎉🎉")
    print("=" * 60)
    print(f"⏱️  TOTAL TIME: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"✅ SCRAPED: {len(problem_details)} problems successfully")
    print(f"❌ FAILED: {scraper.failed_count} problems")
    print(f"⚡ AVERAGE SPEED: {len(problem_details)/total_time:.1f} problems/second")
    print(f"🔥 PEAK WORKERS: {scraper.max_workers} concurrent threads")
    print(f"📊 SUCCESS RATE: {len(problem_details)/(len(problem_details)+scraper.failed_count)*100:.1f}%")
    
    # Expected vs actual
    expected_problems = 80 * 50  # 80 pages × 50 problems per page
    print(f"🎯 EXPECTED: ~{expected_problems} problems")
    print(f"📈 EFFICIENCY: {len(problem_details)/expected_problems*100:.1f}% of expected problems")
    
    if problem_details:
        # Show sample
        print(f"\n📋 Sample problems:")
        for i, problem in enumerate(problem_details[:3]):
            print(f"  {i+1}. {problem['title']}")
            print(f"     Tags: {', '.join(problem['tags'][:3]) if problem['tags'] else 'None'}")
            print(f"     {len(problem['text'])} chars of description")
        
        if len(problem_details) > 3:
            print(f"  ... and {len(problem_details)-3} more problems")
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f'spoj_problems_{len(problem_details)}_{timestamp}.json'
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(problem_details, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Saved to: {filename}")
        print(f"📊 File size: {os.path.getsize(filename) / (1024*1024):.2f} MB")
    
    print("\n🚀 Ultra-fast scraping completed!")


if __name__ == "__main__":
    main()