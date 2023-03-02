import scrapy


class GamesSpider(scrapy.Spider):
    name = 'games'
    allowed_domains = ['store.steampowered.com']
    start_urls = ['http://store.steampowered.com/']

    def parse(self, response):
        pass
