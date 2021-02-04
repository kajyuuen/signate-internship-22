import collections
from html.parser import HTMLParser

from nyaggle.feature_store import cached_feature
import pandas as pd

class TagCounter(HTMLParser):        
    def __init__(self):
        super().__init__()
        self.tag_counter = collections.defaultdict(int)

    def handle_starttag(self, tag, attrs):
        self.tag_counter[tag] += 1

    def get_counter(self):
        return self.tag_counter

@cached_feature("html_tag_all_count", "features")
def create_html_tag_all_count_feature(df):
    all_uniq_tags = {'i', 'param', 'button', 'figcaption', 
                    'figure', 'video', 'polygon', 'audio', 
                    'source', 'use', 'g', 'embed', 'h1', 
                    'style', 'img', 'li', 'animate', 'br', 
                    'iframe', 'time', 'track', 'ol', 'a', 
                    'h2', 'ul', 'p', 'span', 'svg', 'div', 'h3', 'input'}
    dict_count_tags = { "Count_html_" + tag: [] for tag in all_uniq_tags }

    for html_content in df["html_content"]:
        parser = TagCounter()
        parser.feed(html_content)
        counter = parser.get_counter()
        for tag in all_uniq_tags:
            dict_count_tags["Count_html_" + tag].append(counter[tag])
    feat = pd.DataFrame.from_dict(dict_count_tags, orient='index').T

    feat["link_count"] = df["html_content"].str.count("http://")
    feat["enter_count"] = df["html_content"].str.count("\n")

    return feat
