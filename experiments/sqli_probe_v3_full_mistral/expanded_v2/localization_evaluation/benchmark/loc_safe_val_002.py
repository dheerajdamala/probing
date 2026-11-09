def fetch_by_numeric_id_2(conn, raw_id):
    safe_id = int(raw_id)
    return conn.execute(f"SELECT * FROM products WHERE id = {safe_id}").fetchone()
