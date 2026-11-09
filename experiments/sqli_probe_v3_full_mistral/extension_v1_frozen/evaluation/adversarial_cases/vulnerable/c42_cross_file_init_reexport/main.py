from database import run_sql
def handle():
    val = request.args['code']
    sql = f"SELECT * FROM codes WHERE c = '{val}'"
    run_sql(sql)
