from db_ops import execute_custom_8

def handle_request_8(req, payload):
    keyword = payload.get('q')
    return execute_custom_8(keyword)
