"""
Initialise the PostgreSQL database schema.

Run once before the first pipeline execution:
  python scripts/init_db.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models.database import init_db

if __name__ == "__main__":
    print("Initialising database schema …")
    init_db()
    print("Done.")
