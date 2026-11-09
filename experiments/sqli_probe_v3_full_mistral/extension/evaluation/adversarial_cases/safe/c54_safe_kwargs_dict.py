def execute_with_dict(val):
    query = 'SELECT * FROM metrics WHERE metric_name = :m'
    cursor.execute(query, {'m': val})
def api(): execute_with_dict(request.args['m'])
