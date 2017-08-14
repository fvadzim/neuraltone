from peewee import *

database = MySQLDatabase('fin', **{'user': 'mysql', 'password': 'mysql'})

class UnknownField(object):
    def __init__(self, *_, **__): pass

class BaseModel(Model):
    class Meta:
        database = database

class ActiveAdminComments(BaseModel):
    author = IntegerField(db_column='author_id', null=True)
    author_type = CharField(null=True)
    body = TextField(null=True)
    created_at = DateTimeField(null=True)
    namespace = CharField(index=True, null=True)
    resource = CharField(db_column='resource_id')
    resource_type = CharField()
    updated_at = DateTimeField(null=True)

    class Meta:
        db_table = 'active_admin_comments'
        indexes = (
            (('author_type', 'author'), False),
            (('resource_type', 'resource'), False),
        )

class AdminUsers(BaseModel):
    created_at = DateTimeField()
    current_sign_in_at = DateTimeField(null=True)
    current_sign_in_ip = CharField(null=True)
    email = CharField(unique=True)
    encrypted_password = CharField()
    last_sign_in_at = DateTimeField(null=True)
    last_sign_in_ip = CharField(null=True)
    remember_created_at = DateTimeField(null=True)
    reset_password_sent_at = DateTimeField(null=True)
    reset_password_token = CharField(null=True, unique=True)
    sign_in_count = IntegerField()
    updated_at = DateTimeField()

    class Meta:
        db_table = 'admin_users'

class Resources(BaseModel):
    created_at = DateTimeField(null=True)
    domain = CharField()
    module = CharField()
    updated_at = DateTimeField(null=True)
    url = CharField()

    class Meta:
        db_table = 'resources'

class Articles(BaseModel):
    created_at = DateTimeField(null=True)
    error_description = TextField(null=True)
    http_response_status = IntegerField(null=True)
    posted_at = DateTimeField(null=True)
    resource = ForeignKeyField(db_column='resource_id', rel_model=Resources, to_field='id')
    state = CharField()
    static_processed = IntegerField()
    updated_at = DateTimeField(null=True)
    url = CharField()
    with_api = IntegerField()

    class Meta:
        db_table = 'articles'
        indexes = (
            (('resource', 'url'), True),
        )

class Authors(BaseModel):
    created_at = DateTimeField(null=True)
    email = CharField(null=True)
    facebook = CharField(null=True)
    name = CharField()
    resource = ForeignKeyField(db_column='resource_id', rel_model=Resources, to_field='id')
    twitter = CharField(null=True)
    updated_at = DateTimeField(null=True)

    class Meta:
        db_table = 'authors'

class Categories(BaseModel):
    created_at = DateTimeField(null=True)
    name = CharField()
    updated_at = DateTimeField(null=True)

    class Meta:
        db_table = 'categories'

class Publications(BaseModel):
    article = ForeignKeyField(db_column='article_id', rel_model=Articles, to_field='id', unique=True)
    authors_text = CharField(null=True)
    category = ForeignKeyField(db_column='category_id', null=True, rel_model=Categories, to_field='id')
    category_text = CharField(null=True)
    content = TextField(null=True)
    created_at = DateTimeField(null=True)
    image_count = IntegerField(null=True)
    internal = CharField(db_column='internal_id', null=True)
    is_text = IntegerField(null=True)
    is_weekend = IntegerField(null=True)
    keywords_text = CharField(null=True)
    lead = CharField(null=True)
    link_count = IntegerField(null=True)
    meta_tags = CharField(null=True)
    posted_at = DateTimeField(null=True)
    posted_weekday = IntegerField(null=True)
    resource = ForeignKeyField(db_column='resource_id', null=True, rel_model=Resources, to_field='id')
    slug = CharField(null=True)
    time_delta = IntegerField(null=True)
    title = CharField(null=True)
    top_post = IntegerField(null=True)
    updated_at = DateTimeField(null=True)
    url = CharField(null=True)
    video_count = IntegerField(null=True)

    class Meta:
        db_table = 'publications'

class AuthorsPublications(BaseModel):
    author = ForeignKeyField(db_column='author_id', null=True, rel_model=Authors, to_field='id')
    created_at = DateTimeField(null=True)
    publication = ForeignKeyField(db_column='publication_id', null=True, rel_model=Publications, to_field='id')
    updated_at = DateTimeField(null=True)

    class Meta:
        db_table = 'authors_publications'

class Banks(BaseModel):
    name = CharField()

    class Meta:
        db_table = 'banks'

class Configs(BaseModel):
    ahhhhh = IntegerField(null=True)
    comments = IntegerField(null=True)
    created_at = DateTimeField(null=True)
    cute = IntegerField(null=True)
    dislikes = IntegerField(null=True)
    ew = IntegerField(null=True)
    facebook_comments = IntegerField(null=True)
    facebook_reposts = IntegerField(null=True)
    fail = IntegerField(null=True)
    likes = IntegerField(null=True)
    lol = IntegerField(null=True)
    omg = IntegerField(null=True)
    reposts = IntegerField(null=True)
    resource = ForeignKeyField(db_column='resource_id', rel_model=Resources, to_field='id')
    updated_at = DateTimeField(null=True)
    views = IntegerField(null=True)
    win = IntegerField(null=True)
    wtf = IntegerField(null=True)
    yaaass = IntegerField(null=True)

    class Meta:
        db_table = 'configs'

class Credits(BaseModel):
    bank = ForeignKeyField(db_column='bank_id', rel_model=Banks, to_field='id')
    name = CharField()
    rate = FloatField()

    class Meta:
        db_table = 'credits'

class Revisions(BaseModel):
    article = ForeignKeyField(db_column='article_id', rel_model=Articles, to_field='id')
    created_at = DateTimeField(null=True)
    dynamic_processed = IntegerField()
    updated_at = DateTimeField(null=True)

    class Meta:
        db_table = 'revisions'

class DynamicMetrics(BaseModel):
    ahhhhh = IntegerField(null=True)
    comments = IntegerField(null=True)
    created_at = DateTimeField(null=True)
    cute = IntegerField(null=True)
    dislikes = IntegerField(null=True)
    ew = IntegerField(null=True)
    facebook_shares = IntegerField(null=True)
    fail = IntegerField(null=True)
    google_shares = IntegerField(null=True)
    likes = IntegerField(null=True)
    lol = IntegerField(null=True)
    ok_shares = IntegerField(null=True)
    omg = IntegerField(null=True)
    reposts = IntegerField(null=True)
    revision = ForeignKeyField(db_column='revision_id', rel_model=Revisions, to_field='id')
    twitter_shares = IntegerField(null=True)
    updated_at = DateTimeField(null=True)
    views = IntegerField(null=True)
    vk_shares = IntegerField(null=True)
    win = IntegerField(null=True)
    wtf = IntegerField(null=True)
    yaaass = IntegerField(null=True)

    class Meta:
        db_table = 'dynamic_metrics'

class SocialNetworks(BaseModel):
    created_at = DateTimeField(null=True)
    name = CharField()
    updated_at = DateTimeField(null=True)

    class Meta:
        db_table = 'social_networks'

class ExternalMetrics(BaseModel):
    comments = IntegerField(null=True)
    created_at = DateTimeField(null=True)
    is_valid = IntegerField()
    likes = IntegerField(null=True)
    reposts = IntegerField(null=True)
    revision = ForeignKeyField(db_column='revision_id', rel_model=Revisions, to_field='id')
    social_network = ForeignKeyField(db_column='social_network_id', rel_model=SocialNetworks, to_field='id')
    updated_at = DateTimeField(null=True)
    views = IntegerField(null=True)

    class Meta:
        db_table = 'external_metrics'

class Factors(BaseModel):
    average_sentence_length = FloatField(null=True)
    average_token_length = FloatField(null=True)
    average_token_length_syllables = FloatField(null=True)
    avg_negative_polarity = FloatField(null=True)
    avg_positive_polarity = FloatField(null=True)
    factor_type = IntegerField()
    global_negative_polarity = FloatField(null=True)
    global_neutral_polarity = FloatField(null=True)
    global_positive_polarity = FloatField(null=True)
    global_rate_negative_words = FloatField(null=True)
    global_rate_positive_words = FloatField(null=True)
    global_sentiment_polarity = FloatField(null=True)
    global_subjectivity = FloatField(null=True)
    lda = TextField()
    lda_top_topic = CharField(null=True)
    max_negative_polarity = FloatField(null=True)
    max_positive_polarity = FloatField(null=True)
    min_negative_polarity = FloatField(null=True)
    min_positive_polarity = FloatField(null=True)
    most_common_non_stop_words = CharField(null=True)
    n_non_stop_unique_tokens = IntegerField(null=True)
    n_non_stop_words = IntegerField(null=True)
    n_sentences = IntegerField(null=True)
    n_syllables = IntegerField(null=True)
    n_symbols = IntegerField(null=True)
    n_symbols_no_space = IntegerField(null=True)
    n_tokens_content = IntegerField(null=True)
    n_unique_tokens = IntegerField(null=True)
    publication = ForeignKeyField(db_column='publication_id', rel_model=Publications, to_field='id')
    rate_negative_words = FloatField(null=True)
    rate_positive_words = FloatField(null=True)

    class Meta:
        db_table = 'factors'

class FactorsKeywords(BaseModel):
    publication = ForeignKeyField(db_column='publication_id', rel_model=Publications, to_field='id')
    rake_keyphrase = CharField()
    rake_similarity_0 = FloatField()
    rake_similarity_1 = FloatField()
    textrank_keywords = CharField()
    textrank_similarity_0 = FloatField()
    textrank_similarity_1 = FloatField()
    tfidf_keyphrase = CharField()
    tfidf_keywords = CharField()
    tfidf_similarity_0 = FloatField()
    tfidf_similarity_1 = FloatField()

    class Meta:
        db_table = 'factors_keywords'

class Includes(BaseModel):
    entity = IntegerField(db_column='entity_id', null=True)
    resource = ForeignKeyField(db_column='resource_id', rel_model=Resources, to_field='id')
    type = CharField()
    count = IntegerField()

    class Meta:
        db_table = 'includes'

class Keywords(BaseModel):
    created_at = DateTimeField(null=True)
    name = CharField()
    updated_at = DateTimeField(null=True)

    class Meta:
        db_table = 'keywords'

class KeywordsPublications(BaseModel):
    created_at = DateTimeField(null=True)
    keyword = ForeignKeyField(db_column='keyword_id', null=True, rel_model=Keywords, to_field='id')
    publication = ForeignKeyField(db_column='publication_id', null=True, rel_model=Publications, to_field='id')
    updated_at = DateTimeField(null=True)

    class Meta:
        db_table = 'keywords_publications'

class KeywordsStats(BaseModel):
    frequency = FloatField()
    keyword = ForeignKeyField(db_column='keyword_id', rel_model=Keywords, to_field='id')
    resource = ForeignKeyField(db_column='resource_id', rel_model=Resources, to_field='id')

    class Meta:
        db_table = 'keywords_stats'

class Mentions(BaseModel):
    article = ForeignKeyField(db_column='article_id', null=True, rel_model=Articles, to_field='id')
    credit = ForeignKeyField(db_column='credit_id', null=True, rel_model=Credits, to_field='id')
    resource = ForeignKeyField(db_column='resource_id', null=True, rel_model=Resources, to_field='id')

    class Meta:
        db_table = 'mentions'

class SchemaMigrations(BaseModel):
    version = CharField(primary_key=True)

    class Meta:
        db_table = 'schema_migrations'

class StaticMetrics(BaseModel):
    created_at = DateTimeField(null=True)
    is_text = IntegerField(null=True)
    posted_weekday = IntegerField(null=True)
    publication = ForeignKeyField(db_column='publication_id', rel_model=Publications, to_field='id')
    time_delta = IntegerField(null=True)
    updated_at = DateTimeField(null=True)

    class Meta:
        db_table = 'static_metrics'

class Tonalities(BaseModel):
    article = ForeignKeyField(db_column='article_id', rel_model=Articles, to_field='id')
    bank = ForeignKeyField(db_column='bank_id', rel_model=Banks, to_field='id')
    posted_at = DateTimeField(null=True)
    tone = FloatField()

    class Meta:
        db_table = 'tonalities'

class TopKeywords(BaseModel):
    keywords = CharField()
    resource = ForeignKeyField(db_column='resource_id', rel_model=Resources, to_field='id')

    class Meta:
        db_table = 'top_keywords'

class Users(BaseModel):
    confirmation_sent_at = DateTimeField(null=True)
    confirmation_token = CharField(null=True, unique=True)
    confirmed_at = DateTimeField(null=True)
    created_at = DateTimeField()
    current_sign_in_at = DateTimeField(null=True)
    current_sign_in_ip = CharField(null=True)
    email = CharField(unique=True)
    encrypted_password = CharField()
    is_active = IntegerField(null=True)
    last_sign_in_at = DateTimeField(null=True)
    last_sign_in_ip = CharField(null=True)
    remember_created_at = DateTimeField(null=True)
    reset_password_sent_at = DateTimeField(null=True)
    reset_password_token = CharField(null=True, unique=True)
    sign_in_count = IntegerField()
    unconfirmed_email = CharField(null=True)
    updated_at = DateTimeField()

    class Meta:
        db_table = 'users'

