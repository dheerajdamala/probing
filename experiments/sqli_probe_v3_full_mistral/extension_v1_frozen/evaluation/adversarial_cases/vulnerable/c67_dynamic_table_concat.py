def fetch_from_table(tbl):
    q = "SELECT * FROM " + tbl
    cursor.execute(q)
def handle(): fetch_from_table(request.args['t'])
