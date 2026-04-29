with open('app/api.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('_MARKET_INDICES_CACHE_TTL = 60', '_MARKET_INDICES_CACHE_TTL = 3')
content = content.replace('_MARKET_INDEX_DETAIL_CACHE_TTL = 60', '_MARKET_INDEX_DETAIL_CACHE_TTL = 3')
content = content.replace('_MARKET_INDEX_QUOTE_CACHE_TTL = 45', '_MARKET_INDEX_QUOTE_CACHE_TTL = 3')
content = content.replace('_MARKET_INDEX_INTRADAY_CACHE_TTL = 45', '_MARKET_INDEX_INTRADAY_CACHE_TTL = 3')

with open('app/api.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("TTLs updated to 3 seconds")
