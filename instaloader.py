from datetime import datetime
from itertools import dropwhile, takewhile

import instaloader

L = instaloader.Instaloader()

posts = L.get_hashtag_posts('immigration')
# or
# posts = instaloader.Profile.from_username(L.context, PROFILE).get_posts()

SINCE = datetime(2020, 2, 20)
UNTIL = datetime(2020, 3, 10)
filtered_posts = filter(lambda p: SINCE <= p.date <= UNTIL, posts)

for post in filtered_posts:
    print(post.date)
    L.download_post(post, '#immigration')