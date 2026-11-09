def query_limit(lim):
    query = f"SELECT * FROM items LIMIT {lim}"
    cursor.execute(query)
def handle(): query_limit(request.args['limit'])
