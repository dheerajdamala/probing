class BaseDB:
    def run_query(self, query):
        cursor.execute(query)
class UserDB(BaseDB):
    def find_user(self, name):
        self.run_query(f"SELECT * FROM users WHERE name = '{name}'")
def api():
    UserDB().find_user(request.args['name'])
