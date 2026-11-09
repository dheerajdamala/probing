ALLOWED_COLUMNS = ['date', 'amount']
def sort_finance(col):
    if col in ALLOWED_COLUMNS:
        cursor.execute(f"SELECT * FROM txs ORDER BY {col}")
def handle(): sort_finance(request.args['c'])
