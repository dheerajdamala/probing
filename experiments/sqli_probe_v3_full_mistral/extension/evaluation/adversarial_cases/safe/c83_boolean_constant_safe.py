def bool_constant_flow():
    is_admin = True
    query = f"SELECT * FROM users WHERE admin = {is_admin}"
    cursor.execute(query)
