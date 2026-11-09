def lookup_user():
    cursor.execute(f"SELECT * FROM users WHERE id = '{request.args.get('id')}'")
