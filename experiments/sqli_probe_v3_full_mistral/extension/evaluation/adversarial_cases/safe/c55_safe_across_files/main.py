from db import run_param
def handle():
    uid = request.args['id']
    run_param('SELECT * FROM users WHERE id = %s', (uid,))
