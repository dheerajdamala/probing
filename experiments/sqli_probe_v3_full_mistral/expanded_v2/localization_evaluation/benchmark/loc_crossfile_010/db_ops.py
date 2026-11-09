import sqlite3

def execute_custom_10(term):
    conn = sqlite3.connect('app.db')
    sql = f"SELECT * FROM data WHERE title LIKE '%{term}%'"
    return conn.execute(sql).fetchall()
