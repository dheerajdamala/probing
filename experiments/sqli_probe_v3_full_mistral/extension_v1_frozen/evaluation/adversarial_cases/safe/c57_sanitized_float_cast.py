def filter_price():
    price = float(request.args['max_price'])
    sql = f"SELECT * FROM products WHERE price <= {price}"
    cursor.execute(sql)
