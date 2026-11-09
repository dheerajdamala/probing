from db_ops import execute_custom_4

def handle_request_4(req, payload):
    keyword = payload.get('q')
    return execute_custom_4(keyword)
