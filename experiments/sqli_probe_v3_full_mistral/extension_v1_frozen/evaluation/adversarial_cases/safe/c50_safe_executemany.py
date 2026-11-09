def safe_batch(items):
    rows = [(i,) for i in items]
    cursor.executemany('INSERT INTO queue VALUES (%s)', rows)
def handle(): safe_batch(request.args.getlist('item'))
