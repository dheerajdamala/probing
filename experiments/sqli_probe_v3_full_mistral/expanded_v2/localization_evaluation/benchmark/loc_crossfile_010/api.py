from db_ops import execute_custom_10

def handle_request_10(req, payload):
    keyword = payload.get('q')
    return execute_custom_10(keyword)
