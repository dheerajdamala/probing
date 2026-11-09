VALID_STATUS = ['OPEN', 'CLOSED']
def filter_status(s):
    if s not in VALID_STATUS:
        return
    cursor.execute(f"SELECT * FROM tickets WHERE status = '{s}'")
def handle(): filter_status(request.args['s'])
