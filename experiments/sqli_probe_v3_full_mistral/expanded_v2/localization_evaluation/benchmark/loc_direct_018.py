def get_account_balance_18(db, account_id):
    # Line 2: String concatenation
    query = "SELECT balance FROM accounts WHERE acc_id = '" + str(account_id) + "'"
    cursor = db.cursor()
    cursor.execute(query)
    return cursor.fetchone()
