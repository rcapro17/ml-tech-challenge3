CREATE TABLE series (
  code       VARCHAR(50) PRIMARY KEY,
  source     VARCHAR(50),
  name       VARCHAR(255),
  frequency  VARCHAR(50),
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE observations (
  code  VARCHAR(50) NOT NULL,
  ts    DATE        NOT NULL,
  value NUMERIC     NOT NULL,
  PRIMARY KEY (code, ts)
);
CREATE INDEX idx_obs_series_ts ON observations (code, ts);
