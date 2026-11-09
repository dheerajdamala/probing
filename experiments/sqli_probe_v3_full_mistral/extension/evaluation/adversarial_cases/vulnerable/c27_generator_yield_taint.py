def item_generator():
    yield request.args['id']
def run():
    for item in item_generator():
        sql = f"SELECT * FROM queue WHERE id = {item}"
        cursor.execute(sql)
