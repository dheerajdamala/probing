def sort_order(direction):
    if direction == 'asc' or direction == 'desc':
        cursor.execute(f"SELECT * FROM items ORDER BY id {direction}")
def handle(): sort_order(request.args['dir'])
