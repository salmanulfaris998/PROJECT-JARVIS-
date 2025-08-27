#!/usr/bin/env python3
"""
JARVIS Database Reader - Read SQLite logs properly
"""

import sqlite3
from pathlib import Path
from datetime import datetime

def read_jarvis_databases():
    """Read all JARVIS databases and display contents"""
    
    log_dir = Path('logs')
    if not log_dir.exists():
        print("❌ Logs directory not found")
        return
    
    # Find all .db files
    db_files = list(log_dir.glob('*.db'))
    
    if not db_files:
        print("❌ No database files found")
        return
    
    print("🔍 JARVIS Database Analysis")
    print("=" * 50)
    
    for db_file in db_files:
        print(f"\n📊 Database: {db_file.name}")
        print("-" * 30)
        
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            if not tables:
                print("   ⚠️ No tables found")
                continue
            
            for table in tables:
                table_name = table[0]
                print(f"\n📋 Table: {table_name}")
                
                # Get table schema
                cursor.execute(f"PRAGMA table_info({table_name});")
                columns = cursor.fetchall()
                print("   Columns:", [col[1] for col in columns])
                
                # Get row count
                cursor.execute(f"SELECT COUNT(*) FROM {table_name};")
                count = cursor.fetchone()[0]
                print(f"   Rows: {count}")
                
                # Show last 5 rows
                if count > 0:
                    cursor.execute(f"SELECT * FROM {table_name} ORDER BY timestamp DESC LIMIT 5;")
                    rows = cursor.fetchall()
                    print("   Recent entries:")
                    for row in rows:
                        print(f"     {row}")
                else:
                    print("   (No data)")
            
            conn.close()
            
        except Exception as e:
            print(f"   ❌ Error reading {db_file.name}: {str(e)}")
    
    print("\n✅ Database analysis complete!")

if __name__ == '__main__':
    read_jarvis_databases()
