def fetch_orders_by_customer_3(conn, cust_id):
    # Line 2: Legacy % formatting
    stmt = "SELECT * FROM orders WHERE customer_id = '%s'" % cust_id
    return conn.execute(stmt).fetchall()
