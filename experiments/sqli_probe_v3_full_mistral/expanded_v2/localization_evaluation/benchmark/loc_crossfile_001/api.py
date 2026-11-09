from db_ops import execute_custom_1

def handle_request_1(req, payload):
    keyword = payload.get('q')
    return execute_custom_1(keyword)
