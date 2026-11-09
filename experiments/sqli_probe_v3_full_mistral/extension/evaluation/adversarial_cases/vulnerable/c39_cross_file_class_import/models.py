class AccountRepository:
    def execute(self, sql):
        cursor.execute(sql)
    def find_account(self, uid):
        self.execute(f"SELECT * FROM accounts WHERE id = '{uid}'")
