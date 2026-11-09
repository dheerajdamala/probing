def exec_request(req):
    cursor.execute(f"SELECT * FROM store WHERE k = '{req.key}'")
