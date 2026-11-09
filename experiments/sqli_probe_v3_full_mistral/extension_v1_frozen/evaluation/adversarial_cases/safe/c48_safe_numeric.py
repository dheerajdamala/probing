def safe_numeric(tok):
    cursor.execute('SELECT * FROM tokens WHERE t = :1', (tok,))
def handle(): safe_numeric(request.args['t'])
