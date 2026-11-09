import re
def query_numeric(raw):
    if re.match(r'^\d+$', raw):
        cursor.execute(f"SELECT * FROM table WHERE id = {raw}")
def handle(): query_numeric(request.args['id'])
