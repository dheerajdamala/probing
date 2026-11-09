def build_query_2(filter_val):
    return f"SELECT * FROM records WHERE status = '{filter_val}'"

def run_search_2(conn, user_tag):
    q = build_query_2(user_tag)
    return conn.execute(q).fetchall()
