def build_multi_filter_9(cursor, filters):
    query = "SELECT * FROM inventory WHERE 1=1"
    for key, val in filters.items():
        if key == 'status':
            query += " AND status = '" + val + "'"
    cursor.execute(query)
    return cursor.fetchall()
