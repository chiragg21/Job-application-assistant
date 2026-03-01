import sqlite3
import json

def get_database_schema(db_path: str):
    """
    Connects to the SQLite database and returns the full schema 
    (Table names, columns, types, and constraints).
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 1. Get all table names (excluding internal sqlite tables)
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
        tables = cursor.fetchall()

        schema_report = {}

        for table_name in tables:
            name = table_name[0]
            
            # 2. Get column details: cid, name, type, notnull, dflt_value, pk
            cursor.execute(f"PRAGMA table_info('{name}');")
            columns = cursor.fetchall()
            
            # 3. Get the original CREATE statement (useful for foreign keys/constraints)
            cursor.execute(f"SELECT sql FROM sqlite_master WHERE name='{name}';")
            create_statement = cursor.fetchone()[0]

            schema_report[name] = {
                "columns": [
                    {
                        "name": col[1],
                        "type": col[2],
                        "not_null": bool(col[3]),
                        "default_value": col[4],
                        "primary_key": bool(col[5])
                    } for col in columns
                ],
                "raw_sql": create_statement
            }

        conn.close()
        return schema_report

    except Exception as e:
        return {"error": str(e)}

# Usage
if __name__ == "__main__":
    # Update this path to your actual .db file location
    db_file = "data/app.db" 
    full_schema = get_database_schema(db_file)
    
    # Print as formatted JSON for easy reading
    with open("output.json", 'w') as f:
        json.dump(full_schema, f, indent=2)