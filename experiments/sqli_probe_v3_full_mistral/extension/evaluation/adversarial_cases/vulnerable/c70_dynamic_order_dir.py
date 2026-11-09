def sort_direction(direction):
    sql = f"SELECT * FROM logs ORDER BY ts {direction}"
    cursor.execute(sql)
def handle(): sort_direction(request.args['dir'])
