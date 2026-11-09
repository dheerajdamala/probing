from db_ops import execute_custom_14

def handle_request_14(req, payload):
    keyword = payload.get('q')
    return execute_custom_14(keyword)
