def find_records(table, filter_arg):
    cursor.execute(f"SELECT * FROM {table} WHERE val = '{filter_arg}'")
def route():
    find_records('users', request.args['f'])
