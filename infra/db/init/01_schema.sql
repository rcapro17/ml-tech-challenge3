CREATE TABLE IF NOT EXISTS series (
  code        VARCHAR(20) PRIMARY KEY,
  source      VARCHAR(50) NOT NULL,
  name        VARCHAR(200),
  frequency   VARCHAR(20)
);

CREATE TABLE IF NOT EXISTS observations (
  series_code VARCHAR(20) NOT NULL,
  ts          DATE NOT NULL,
  value       NUMERIC,
  ingested_at TIMESTAMP DEFAULT NOW(),
  PRIMARY KEY (series_code, ts),
  CONSTRAINT fk_series FOREIGN KEY (series_code) REFERENCES series(code)
);

CREATE INDEX IF NOT EXISTS idx_obs_series_ts ON observations (series_code, ts);
