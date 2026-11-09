class SafeRepository:
    def find(self, key):
        cursor.execute('SELECT * FROM data WHERE k = %s', (key,))
def run():
    SafeRepository().find(request.args['k'])
