MAPPING = {'name': 'username', 'email': 'user_email'}
def query_mapped(field):
    col = MAPPING.get(field, 'id')
    cursor.execute(f"SELECT {col} FROM accounts")
def handle(): query_mapped(request.args['f'])
