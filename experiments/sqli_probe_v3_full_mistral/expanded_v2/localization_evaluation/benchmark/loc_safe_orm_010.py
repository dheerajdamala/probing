def query_session_user_10(session, username):
    stmt = select(User).where(User.username == username)
    return session.scalars(stmt).all()
