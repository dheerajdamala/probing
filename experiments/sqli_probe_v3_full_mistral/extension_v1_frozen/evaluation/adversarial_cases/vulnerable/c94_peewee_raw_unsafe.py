def raw_peewee(code):
    db.execute_sql(f"SELECT * FROM inventory WHERE code = '{code}'")
def api(): raw_peewee(request.args['c'])
