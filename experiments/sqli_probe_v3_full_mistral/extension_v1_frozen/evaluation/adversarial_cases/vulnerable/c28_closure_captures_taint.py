def outer_handler(user_input):
    def inner_executor():
        query = f"SELECT * FROM users WHERE name = '{user_input}'"
        cursor.execute(query)
    inner_executor()
def route():
    outer_handler(request.args['name'])
