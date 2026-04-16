import trafilatura
from readability import Document
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_article(url, timeout=15):
    """
    Extracts article content from a URL using trafilatura with a fallback to readability-lxml.
    
    Returns:
        dict: {
            'title': str,
            'clean_text': str,
            'source_domain': str,
            'extraction_method': str,
            'warnings': list,
            'input_url': str
        }
    """
    results = {
        'title': None,
        'clean_text': None,
        'source_domain': None,
        'extraction_method': 'failed',
        'warnings': [],
        'input_url': url
    }
    
    try:
        domain = urlparse(url).netloc
        results['source_domain'] = domain
    except Exception as e:
        results['warnings'].append(f"Could not parse domain: {str(e)}")

    # 1. Try Trafilatura
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            content = trafilatura.extract(downloaded, include_comments=False, include_tables=True, no_fallback=False)
            metadata = trafilatura.extract_metadata(downloaded)
            
            if content and len(content.strip()) > 200:
                results['clean_text'] = content
                results['title'] = metadata.title if metadata and metadata.title else None
                results['extraction_method'] = 'trafilatura'
                
                # Check for boilerplate in title/text
                if results['title'] and ("Access Denied" in results['title'] or "Robot Check" in results['title']):
                    results['warnings'].append("Possible bot detection or access denial.")
                
                return _validate_and_cleanup(results)
    except Exception as e:
        results['warnings'].append(f"Trafilatura failed: {str(e)}")

    # 2. Fallback to Readability + BeautifulSoup
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        doc = Document(response.text)
        summary_html = doc.summary()
        title = doc.title()
        
        soup = BeautifulSoup(summary_html, 'lxml')
        # Remove common junk if it survived
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()
            
        clean_text = soup.get_text(separator='\n').strip()
        
        if len(clean_text) > 200:
            results['clean_text'] = clean_text
            results['title'] = title
            results['extraction_method'] = 'readability-fallback'
            return _validate_and_cleanup(results)
            
    except Exception as e:
        results['warnings'].append(f"Readability fallback failed: {str(e)}")

    return results

def _validate_and_cleanup(results):
    """Refine extracted text and add warnings for quality issues."""
    text = results['clean_text']
    if not text:
        return results
    
    # 1. Basic Cleaning
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove typical cookie/privacy fragments
    boilerplate_patterns = [
        r"This website uses cookies",
        r"Accept all cookies",
        r"Sign up for our newsletter",
        r"Follow us on social media",
        r"Copyright © \d{4}",
        r"All rights reserved"
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    results['clean_text'] = text.strip()
    
    # 2. Validation
    if len(results['clean_text']) < 300:
        results['warnings'].append("Extracted text is very short. It might be a summary or partial content.")
    
    # Ratio of uppercase might indicate noise or ads
    upper_ratio = sum(1 for c in results['clean_text'] if c.isupper()) / len(results['clean_text'])
    if upper_ratio > 0.3:
        results['warnings'].append("High uppercase ratio detected. Might contain noisy text or ads.")

    return results

if __name__ == "__main__":
    # Test with a known URL
    test_url = "https://www.cnn.com/2024/04/15/politics/trump-hush-money-trial-day-1/index.html"
    print(f"Testing extraction for: {test_url}")
    res = extract_article(test_url)
    print(f"Method: {res['extraction_method']}")
    print(f"Title: {res['title']}")
    print(f"Warnings: {res['warnings']}")
    print(f"Text snippet: {res['clean_text'][:200]}...")
