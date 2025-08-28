# config.py
import os

# DB_MODE: 'sqlite' (default), 'postgres', 'mongo'
DB_MODE = os.getenv("DB_MODE", "sqlite")

# Example Postgres URL: postgresql+psycopg2://user:pass@localhost:5432/rt_test
POSTGRES_URL = os.getenv("POSTGRES_URL", "")

# MongoDB example: mongodb://localhost:27017/
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017/")

# Random seed
RNG_SEED = int(os.getenv("RNG_SEED", 42))
