def build_query_5(filter_val):
    return f"SELECT * FROM records WHERE status = '{filter_val}'"

def run_search_5(conn, user_tag):
    q = build_query_5(user_tag)
    return conn.execute(q).fetchall()
