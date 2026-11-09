def build_query_1(filter_val):
    return f"SELECT * FROM records WHERE status = '{filter_val}'"

def run_search_1(conn, user_tag):
    q = build_query_1(user_tag)
    return conn.execute(q).fetchall()
