from dto import QueryRequest
from db import exec_request
def run():
    req = QueryRequest(request.args['key'])
    exec_request(req)
