def get_sorted_logs_9(cursor, sort_field):
    ALLOWED = {'time': 'timestamp', 'level': 'log_level', 'id': 'log_id'}
    col = ALLOWED.get(sort_field, 'timestamp')
    cursor.execute(f"SELECT * FROM logs ORDER BY {col} ASC")
    return cursor.fetchall()
