from models import AccountRepository
def get_account():
    repo = AccountRepository()
    repo.find_account(request.args['id'])
