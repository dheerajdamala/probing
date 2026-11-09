def assemble_join(user_input):
    query = "".join(["SELECT * FROM users WHERE name = '", user_input, "'"])
    cursor.execute(query)
