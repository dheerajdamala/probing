def list_user_articles_3(request, author_id):
    articles = Article.objects.filter(author_id=author_id, is_published=True)
    return list(articles.values('id', 'title'))
