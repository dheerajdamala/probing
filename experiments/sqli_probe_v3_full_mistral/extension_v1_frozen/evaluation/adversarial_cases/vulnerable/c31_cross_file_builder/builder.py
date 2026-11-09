def build_user_sql(uid):
    return f"SELECT * FROM users WHERE id = '{uid}'"
