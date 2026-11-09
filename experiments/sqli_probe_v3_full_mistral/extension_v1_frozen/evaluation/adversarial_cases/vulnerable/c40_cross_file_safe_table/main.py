from constants import USER_TABLE
def run():
    uid = request.args['id']
    sql = f"SELECT * FROM {USER_TABLE} WHERE id = '{uid}'"
    cursor.execute(sql)
