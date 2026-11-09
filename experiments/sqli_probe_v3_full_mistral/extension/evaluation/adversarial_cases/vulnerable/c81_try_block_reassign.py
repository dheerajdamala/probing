def try_reassign():
    try:
        q = f"SELECT * FROM logs WHERE l = '{request.args['l']}'"
    except Exception:
        q = 'SELECT 1'
    cursor.execute(q)
