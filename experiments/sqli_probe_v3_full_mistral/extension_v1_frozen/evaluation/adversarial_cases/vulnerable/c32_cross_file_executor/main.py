from db_driver import execute_statement
def handle():
    name = request.args['name']
    query = "SELECT * FROM staff WHERE n = '" + name + "'"
    execute_statement(query)
