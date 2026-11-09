def build_query_6(filter_val):
    return f"SELECT * FROM records WHERE status = '{filter_val}'"

def run_search_6(conn, user_tag):
    q = build_query_6(user_tag)
    return conn.execute(q).fetchall()
