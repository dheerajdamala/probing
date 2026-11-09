def query_session_user_8(session, username):
    stmt = select(User).where(User.username == username)
    return session.scalars(stmt).all()
