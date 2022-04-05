import gzip
import os
import random
import re
import shutil
import tarfile
from glob import glob
from html.parser import HTMLParser

import numpy as np
import requests
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import MultiLabelBinarizer


def get_content(url):
    filename = url.split('/')[-1]

    if not os.path.exists(filename):
        content = requests.get(url).text
        with open(filename, mode='wt', encoding='utf-8') as outfile:
            outfile.write(content)
        return content
    else:
        with open(filename, mode='rt', encoding='utf-8') as infile:
            return infile.read()


def get_imdb(r=None):
    url_train = "http://esuli.it/demo/data/imdb_train.txt"
    url_test = "http://esuli.it/demo/data/imdb_test.txt"

    content = get_content(url_train)

    train_docs = list()
    train_labels = list()
    for line in content.split('\n'):
        if len(line) > 0:
            doc, label = line.strip().split('\t')
            train_docs.append(doc)
            train_labels.append(label)

    shuffled_id = list(range(len(train_docs)))
    if r is None:
        r = random
    r.shuffle(shuffled_id)

    train_docs = [train_docs[i] for i in shuffled_id]
    train_labels = [train_labels[i] for i in shuffled_id]

    content = get_content(url_test)

    test_docs = list()
    test_labels = list()
    for line in content.split('\n'):
        line = line.strip()
        if len(line) > 2:
            doc, label = line.strip().split('\t')
            test_docs.append(doc)
            test_labels.append(label)

    return np.asarray(train_docs), np.asarray(train_labels), np.asarray(test_docs), np.asarray(test_labels)


def get_tng(train_pool_size, r=None):
    tng_dataset = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))

    documents = tng_dataset.data
    class_names = tng_dataset.target_names
    labels = [class_names[i] for i in tng_dataset.target]

    shuffled_id = list(range(len(documents)))
    if r is None:
        r = random
    r.shuffle(shuffled_id)

    train_docs_id = shuffled_id[:train_pool_size]
    test_docs_id = shuffled_id[train_pool_size:]

    train_docs = [documents[doc_id] for doc_id in train_docs_id]
    test_docs = [documents[doc_id] for doc_id in test_docs_id]

    train_labels = [labels[doc_id] for doc_id in train_docs_id]
    test_labels = [labels[doc_id] for doc_id in test_docs_id]

    return np.asarray(train_docs), np.asarray(train_labels), np.asarray(test_docs), np.asarray(test_labels)


def get_filename(url):
    filename = url.split('/')[-1]

    if not os.path.exists(filename):
        with requests.get(url, stream=True) as r:
            with open(filename, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
    return filename


class ReutersParser(HTMLParser):
    def __init__(self, encoding='latin-1'):
        HTMLParser.__init__(self)
        self._reset()
        self.encoding = encoding

    def handle_starttag(self, tag, attrs):
        method = 'start_' + tag
        getattr(self, method, lambda x: None)(attrs)

    def handle_endtag(self, tag):
        method = 'end_' + tag
        getattr(self, method, lambda: None)()

    def _reset(self):
        self.in_title = 0
        self.in_body = 0
        self.in_topics = 0
        self.in_topic_d = 0
        self.title = ""
        self.body = ""
        self.topics = []
        self.topic_d = ""

    def parse(self, fd):
        self.docs = []
        for chunk in fd:
            self.feed(chunk.decode(self.encoding))
            for doc in self.docs:
                yield doc
            self.docs = []
        self.close()

    def handle_data(self, data):
        if self.in_body:
            self.body += data
        elif self.in_title:
            self.title += data
        elif self.in_topic_d:
            self.topic_d += data

    def start_reuters(self, attributes):
        pass

    def end_reuters(self):
        self.body = re.sub(r'\s+', r' ', self.body)
        self.docs.append({'title': self.title,
                          'body': self.body,
                          'topics': self.topics})
        self._reset()

    def start_title(self, attributes):
        self.in_title = 1

    def end_title(self):
        self.in_title = 0

    def start_body(self, attributes):
        self.in_body = 1

    def end_body(self):
        self.in_body = 0

    def start_topics(self, attributes):
        self.in_topics = 1

    def end_topics(self):
        self.in_topics = 0

    def start_d(self, attributes):
        self.in_topic_d = 1

    def end_d(self):
        self.in_topic_d = 0
        self.topics.append(self.topic_d)
        self.topic_d = ""


def stream_reuters_documents(data_path=None):
    DOWNLOAD_URL = ('http://archive.ics.uci.edu/ml/machine-learning-databases/'
                    'reuters21578-mld/reuters21578.tar.gz')

    if data_path is None:
        data_path = '../data/reuters21578'

    os.makedirs(data_path, exist_ok=True)

    archive_path = get_filename(DOWNLOAD_URL)
    tarfile.open(archive_path, 'r:gz').extractall(data_path)

    parser = ReutersParser()
    for filename in glob(os.path.join(data_path, "*.sgm")):
        for doc in parser.parse(open(filename, 'rb')):
            yield doc


def get_reut(r=None, get_label_idx=False):
    topics = {'acq', 'alum', 'austdlr', 'austral', 'barley', 'bfr', 'bop', 'can', 'carcass', 'castor-meal',
              'castor-oil', 'castorseed', 'citruspulp', 'cocoa', 'coconut', 'coconut-oil', 'coffee', 'copper',
              'copra-cake', 'corn', 'corn-oil', 'cornglutenfeed', 'cotton ', 'cotton-meal', 'cotton-oil', 'cottonseed',
              'cpi', 'cpu', 'crude', 'cruzado', 'dfl', 'dkr', 'dlr', 'dmk', 'drachma', 'earn', 'escudo', 'f-cattle',
              'ffr', 'fishmeal', 'flaxseed', 'fuel', 'gas', 'gnp', 'gold', 'grain', 'groundnut', 'groundnut-meal',
              'groundnut-oil', 'heat', 'hk', 'hog', 'housing', 'income', 'instal-debt', 'interest', 'inventories',
              'ipi', 'iron-steel', 'jet', 'jobs', 'l-cattle', 'lead', 'lei', 'lin-meal', 'lin-oil', 'linseed', 'lit',
              'livestock', 'lumber', 'lupin', 'meal-feed', 'mexpeso', 'money-fx', 'money-supply', 'naphtha     ',
              'nat-gas', 'nickel', 'nkr', 'nzdlr', 'oat', 'oilseed', 'orange', 'palladium', 'palm-meal', 'palm-oil',
              'palmkernel', 'peseta', 'pet-chem', 'platinum', 'plywood', 'pork-belly', 'potato', 'propane', 'rand',
              'rape-meal', 'rape-oil', 'rapeseed', 'red-bean', 'reserves', 'retail', 'rice', 'ringgit', 'rubber',
              'rupiah', 'rye', 'saudriyal', 'sfr', 'ship', 'silk', 'silver', 'singdlr', 'skr', 'sorghum', 'soy-meal',
              'soy-oil', 'soybean', 'stg', 'strategic-metal', 'sugar', 'sun-meal', 'sun-oil', 'sunseed', 'tapioca',
              'tea', 'tin', 'trade', 'tung', 'tung-oil', 'veg-oil', 'wheat', 'wool', 'wpi', 'yen', 'zinc'}

    train_docs = list()
    train_labels = list()
    test_docs = list()
    test_labels = list()
    reuters_dataset = stream_reuters_documents()
    size = 9603
    for doc in reuters_dataset:
        if len(train_labels) < size:
            train_docs.append(doc['title'] + '\n' + doc['body'])
            doc_topics = list(set(doc['topics']).intersection(topics))
            train_labels.append(doc_topics)
        else:
            test_docs.append(doc['title'] + '\n' + doc['body'])
            doc_topics = list(set(doc['topics']).intersection(topics))
            test_labels.append(doc_topics)

    shuffled_id = list(range(len(train_docs)))
    if r is None:
        r = random
    r.shuffle(shuffled_id)

    train_docs = [train_docs[i] for i in shuffled_id]
    train_labels = [train_labels[i] for i in shuffled_id]

    binarizer = MultiLabelBinarizer()

    train_labels = binarizer.fit_transform(train_labels)
    test_labels = binarizer.transform(test_labels)

    if get_label_idx:
        return np.asarray(train_docs), train_labels, np.asarray(test_docs), test_labels, list(binarizer.classes_)
    else:
        return np.asarray(train_docs), train_labels, np.asarray(test_docs), test_labels


def get_fine_food(r=None, train_size=25000, test_size=25000):
    url = 'http://snap.stanford.edu/data/finefoods.txt.gz'

    filename = url.split('/')[-1]

    if not os.path.exists(filename):
        stream = requests.get(url, stream=True)
        with open(filename, 'wb') as f:
            for chunk in stream.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
    docs = list()
    labels = list()
    with gzip.open(filename, mode='rt', encoding='utf-8', errors='ignore') as infile:
        for line in infile:
            line = line.strip()
            if line.startswith('review/score:'):
                score = int(line[-3:-2])
                if score >= 4:
                    label = '1'
                else:
                    label = '0'
            if line.startswith('review/text:'):
                text = line[13:]
                docs.append(text)
                labels.append(label)

    if r is None:
        r = random
    pos_ids = [i for i in range(len(labels)) if labels[i] == '1']
    neg_ids = [i for i in range(len(labels)) if labels[i] == '0']

    selected_pos_ids = r.sample(pos_ids, (train_size + test_size) // 2)
    selected_neg_ids = r.sample(neg_ids, (train_size + test_size) // 2)

    train_ids = selected_pos_ids[:train_size // 2] + selected_neg_ids[:train_size // 2]
    test_ids = selected_pos_ids[train_size // 2:] + selected_neg_ids[train_size // 2:]

    r.shuffle(train_ids)
    r.shuffle(test_ids)

    train_docs = [docs[id] for id in train_ids]
    train_labels = [labels[id] for id in train_ids]

    test_docs = [docs[id] for id in test_ids]
    test_labels = [labels[id] for id in test_ids]

    return np.asarray(train_docs), np.asarray(train_labels), np.asarray(test_docs), np.asarray(test_labels)
