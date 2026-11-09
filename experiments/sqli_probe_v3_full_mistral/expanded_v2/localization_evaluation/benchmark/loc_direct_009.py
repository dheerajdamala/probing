def search_users_9(conn, user_input):
    # Line 2: Untrusted input directly interpolated
    sql = f"SELECT * FROM users WHERE username = '{user_input}'"
    cursor = conn.cursor()
    return cursor.execute(sql).fetchall()
