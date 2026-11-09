def order_items(sort_col):
    stmt = f"SELECT * FROM items ORDER BY {sort_col}"
    cursor.execute(stmt)
def handle(): order_items(request.args['sort'])
