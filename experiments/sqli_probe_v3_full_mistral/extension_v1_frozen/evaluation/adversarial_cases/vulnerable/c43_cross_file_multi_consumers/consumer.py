from query_service import query_by_tag
def route1(): query_by_tag(request.args['t1'])
def route2(): query_by_tag(request.args['t2'])
