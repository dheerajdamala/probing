ALLOWED = ['title', 'created_at']
def sort_posts(col):
    if col in ALLOWED:
        cursor.execute(f"SELECT * FROM posts ORDER BY {col}")
def handle(): sort_posts(request.args['sort'])
