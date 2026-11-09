def query_in_clause(id_list):
    placeholders = ', '.join(['%s'] * len(id_list))
    stmt = f'SELECT * FROM items WHERE id IN ({placeholders})'
    cursor.execute(stmt, id_list)
def handle(): query_in_clause(request.args.getlist('ids'))
