class SearchContext:
    def __init__(self, query_term):
        self.term = query_term
    def execute(self):
        cursor.execute(f"SELECT * FROM docs WHERE doc = '{self.term}'")
def search():
    ctx = SearchContext(request.args['q'])
    ctx.execute()
