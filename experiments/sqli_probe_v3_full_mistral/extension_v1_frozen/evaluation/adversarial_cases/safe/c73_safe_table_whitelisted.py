ALLOWED_TABLES = ['users', 'admins']
def fetch_data(t):
    if t in ALLOWED_TABLES:
        cursor.execute(f"SELECT * FROM {t}")
def handle(): fetch_data(request.args['t'])
