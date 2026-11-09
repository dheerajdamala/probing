def auth(user, pwd):
    sql = "SELECT id FROM accounts WHERE u='%s' AND p='%s'" % (user, pwd)
    cursor.execute(sql)
