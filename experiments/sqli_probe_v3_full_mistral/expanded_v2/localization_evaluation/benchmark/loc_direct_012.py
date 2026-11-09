def filter_catalog_12(cursor, category_name):
    # Line 2: .format() method assembly
    raw_sql = "SELECT id, title, price FROM catalog WHERE category = '{}'".format(category_name)
    cursor.execute(raw_sql)
    return cursor.fetchall()
