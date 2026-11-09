def safe_qmark(user_id):
    cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
def handle(): safe_qmark(request.args['id'])
