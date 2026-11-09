def fetch_members(group_id):
    return User.objects.raw(f"SELECT * FROM auth_user WHERE gid = {group_id}")
def view(): fetch_members(request.args['gid'])
