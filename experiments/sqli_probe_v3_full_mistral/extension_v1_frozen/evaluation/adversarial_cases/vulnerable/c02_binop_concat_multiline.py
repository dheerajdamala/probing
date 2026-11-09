def query_products(tag):
    query = (
        "SELECT id, name FROM items WHERE tag = '" + tag + "'"
    )
    db.execute(query)
