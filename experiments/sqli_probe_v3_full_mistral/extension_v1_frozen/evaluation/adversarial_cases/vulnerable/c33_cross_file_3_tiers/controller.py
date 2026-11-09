from service import service_find
def endpoint():
    raw = request.args['filter']
    service_find(raw)
