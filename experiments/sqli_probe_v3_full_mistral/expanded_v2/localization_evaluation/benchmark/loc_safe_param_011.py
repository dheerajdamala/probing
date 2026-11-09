def get_user_profile_11(conn, user_id):
    sql = "SELECT id, username, email FROM users WHERE id = ?"
    cursor = conn.cursor()
    cursor.execute(sql, (user_id,))
    return cursor.fetchone()
