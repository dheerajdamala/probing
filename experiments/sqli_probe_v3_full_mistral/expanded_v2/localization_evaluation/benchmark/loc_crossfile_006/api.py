from db_ops import execute_custom_6

def handle_request_6(req, payload):
    keyword = payload.get('q')
    return execute_custom_6(keyword)
