from db_ops import execute_custom_2

def handle_request_2(req, payload):
    keyword = payload.get('q')
    return execute_custom_2(keyword)
