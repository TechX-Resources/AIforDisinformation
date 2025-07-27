import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


def remove_html_tags(raw_html):
    return BeautifulSoup(raw_html, "html.parser").get_text()

def remove_special_chars(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def remove_stopwords(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in tokens if word.lower() not in stop_words])

def normalize(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def preprocess_text(raw_html):
    text = remove_html_tags(raw_html)
    text = normalize(text)
    text = remove_special_chars(text)
    text = remove_stopwords(text)
    return text

# test
if __name__ == "__main__":
    sample_html = "<html><body><h1>This is a test</h1><p>Hello world!</p></body></html>"
    print(preprocess_text(sample_html))