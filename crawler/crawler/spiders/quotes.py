from pathlib import Path

import scrapy


class QuotesSpider(scrapy.Spider):
    name = 'quotes'
    allowed_domains = ['quotes.toscrape.com']
    start_urls = [
    	'http://quotes.toscrape.com/page/1/',
    	'http://quotes.toscrape.com/page/2/',
    ]
    
    DATA_DIR = "data"
    
    def start_requests(self):
    	Path(self.DATA_DIR).mkdir(parents=True, exist_ok=True)
    	
    	for url in self.start_urls:
    		yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = f"quotes-{page}.html"
        (Path("data") / filename).write_bytes(response.body)
        self.log(f'Saved file {filename}')
    
