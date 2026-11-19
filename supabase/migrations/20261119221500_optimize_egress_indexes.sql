CREATE INDEX IF NOT EXISTS idx_ragfin_json_cache_key_pattern ON ragfin_json_cache (cache_key text_pattern_ops);
CREATE INDEX IF NOT EXISTS idx_instruments_updated_at ON instruments (updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_instrument_aliases_updated_at ON instrument_aliases (updated_at DESC);
DROP INDEX IF EXISTS idx_fund_prices_code_date;
