from flask import Flask, jsonify
import math
import re

from flask import Flask, render_template, request, jsonify, url_for
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, RadioField, SelectMultipleField


def lc_load_vocab():
    vocab = {}
    with open("lc_vocab.txt", "r") as f:
        vocab_terms = f.readlines()
    with open("lc_idf-values.txt", "r") as f:
        idf_values = f.readlines()

    for (term, idf_value) in zip(vocab_terms, idf_values):
        vocab[term.rstrip()] = int(idf_value.rstrip())

    return vocab

def cf_load_vocab():
    vocab = {}
    with open("cf_vocab.txt", "r") as f:
        vocab_terms = f.readlines()
    with open("cf_idf-values.txt", "r") as f:
        idf_values = f.readlines()

    for (term, idf_value) in zip(vocab_terms, idf_values):
        vocab[term.rstrip()] = int(idf_value.rstrip())

    return vocab


def lc_load_document():
    with open("lc_document.txt", "r") as f:
        documents = f.readlines()

    # print('Number of documents: ', len(documents))
    # print('Sample document: ', documents[0])
    return documents

def cf_load_document():
    with open("cf_documents.txt", "r") as f:
        documents = f.readlines()

    # print('Number of documents: ', len(documents))
    # print('Sample document: ', documents[0])
    return documents


def lc_load_inverted_index():
    inverted_index = {}
    with open('lc_inverted_index.txt', 'r') as f:
        inverted_index_terms = f.readlines()

    for row_num in range(0, len(inverted_index_terms), 2):
        term = inverted_index_terms[row_num].strip()
        documents = inverted_index_terms[row_num+1].strip().split()
        inverted_index[term] = documents

    # print('Size of inverted index: ', len(inverted_index))
    return inverted_index

def cf_load_inverted_index():
    inverted_index = {}
    with open('cf_inverted_index.txt', 'r') as f:
        inverted_index_terms = f.readlines()

    for row_num in range(0, len(inverted_index_terms), 2):
        term = inverted_index_terms[row_num].strip()
        documents = inverted_index_terms[row_num+1].strip().split()
        inverted_index[term] = documents

    # print('Size of inverted index: ', len(inverted_index))
    return inverted_index



def lc_load_link_of_qs():
    with open("Scraping/lc_qData/Qlink.txt", "r") as f:
        links = f.readlines()

    return links

def cf_load_link_of_qs():
    with open("Scraping/cf_qData/Qlink.txt", "r") as f:
        links = f.readlines()

    return links

lc_vocab = lc_load_vocab()            # lc_vocab : idf_values
lc_document = lc_load_document()
lc_inverted_index = lc_load_inverted_index()
lc_Qlink = lc_load_link_of_qs()

cf_vocab = cf_load_vocab()            # cf_vocab : idf_values
cf_document = cf_load_document()
cf_inverted_index = cf_load_inverted_index()
cf_Qlink = cf_load_link_of_qs()


def lc_get_tf_dict(term):
    tf_dict = {}
    if term in lc_inverted_index:
        for doc in lc_inverted_index[term]:
            if doc not in tf_dict:
                tf_dict[doc] = 1
            else:
                tf_dict[doc] += 1

    for doc in tf_dict:
        # dividing the freq of the word in doc with the total no of words in doc indexed document
        try:
            tf_dict[doc] /= len(lc_document[int(doc)])
        except (ZeroDivisionError, ValueError, IndexError) as e:
            print(e)
            print(doc)

    return tf_dict

def cf_get_tf_dict(term):
    tf_dict = {}
    if term in cf_inverted_index:
        for doc in cf_inverted_index[term]:
            if doc not in tf_dict:
                tf_dict[doc] = 1
            else:
                tf_dict[doc] += 1

    for doc in tf_dict:
        # dividing the freq of the word in doc with the total no of words in doc indexed document
        try:
            tf_dict[doc] /= len(cf_document[int(doc)])
        except (ZeroDivisionError, ValueError, IndexError) as e:
            print(e)
            print(doc)

    return tf_dict


def lc_get_idf_value(term):
    return math.log((1 + len(lc_document)) / (1 + lc_vocab[term]))

def cf_get_idf_value(term):
    return math.log((1 + len(cf_document)) / (1 + cf_vocab[term]))


def lc_calc_docs_sorted_order(q_terms):
    # will store the doc which can be our ans: sum of tf-idf value of that doc for all the query terms
    potential_docs = {}
    ans = []
    for term in q_terms:
        if (term not in lc_vocab):
            continue

        tf_vals_by_docs = lc_get_tf_dict(term)
        idf_value = lc_get_idf_value(term)

        # print(term, tf_vals_by_docs, idf_value)

        for doc in tf_vals_by_docs:
            if doc not in potential_docs:
                potential_docs[doc] = tf_vals_by_docs[doc]*idf_value
            else:
                potential_docs[doc] += tf_vals_by_docs[doc]*idf_value

        # print(potential_docs)
        # divide the scores of each doc with no of query terms
        for doc in potential_docs:
            potential_docs[doc] /= len(q_terms)

        # sort in dec order acc to values calculated
        potential_docs = dict(
            sorted(potential_docs.items(), key=lambda item: item[1], reverse=True))

        # if no doc found
        if (len(potential_docs) == 0):
            print("No matching question found. Please search with more relevant terms.")

        # Printing ans
        # print("The Question links in Decreasing Order of Relevance are: \n")
        for doc_index in potential_docs:
            # print("Question Link:", Qlink[int(
            #     doc_index) - 1], "\tScore:", potential_docs[doc_index])
            ans.append({"Question Link": lc_Qlink[int(doc_index) - 1][:-2],
                        "Question Name": ((lc_Qlink[int(doc_index) - 1][:-2].replace("https://leetcode.com/problems/", "")).title()).replace('-', ' '),
                        "Score": potential_docs[doc_index]})
    return ans

def cf_calc_docs_sorted_order(q_terms):
    # will store the doc which can be our ans: sum of tf-idf value of that doc for all the query terms
    potential_docs = {}
    ans = []
    for term in q_terms:
        if (term not in cf_vocab):
            continue

        tf_vals_by_docs = cf_get_tf_dict(term)
        idf_value = cf_get_idf_value(term)

        # print(term, tf_vals_by_docs, idf_value)

        for doc in tf_vals_by_docs:
            if doc not in potential_docs:
                potential_docs[doc] = tf_vals_by_docs[doc]*idf_value
            else:
                potential_docs[doc] += tf_vals_by_docs[doc]*idf_value

        # print(potential_docs)
        # divide the scores of each doc with no of query terms
        for doc in potential_docs:
            potential_docs[doc] /= len(q_terms)

        # sort in dec order acc to values calculated
        potential_docs = dict(
            sorted(potential_docs.items(), key=lambda item: item[1], reverse=True))

        # if no doc found
        if (len(potential_docs) == 0):
            print("No matching question found. Please search with more relevant terms.")

        # Printing ans
        # print("The Question links in Decreasing Order of Relevance are: \n")
        for doc_index in potential_docs:
            # print("Question Link:", Qlink[int(
            #     doc_index) - 1], "\tScore:", potential_docs[doc_index])
            ans.append({"Question Link": cf_Qlink[int(doc_index) - 1][:],
                        "Question Name": cf_Qlink[int(doc_index) - 1][:],
                        "Score": potential_docs[doc_index]})
    return ans

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
# query = input('Enter your query: ')
# q_terms = [term.lower() for term in query.strip().split()]

# print(q_terms)
# print(calc_docs_sorted_order(q_terms)[0])
# print(len(calc_docs_sorted_order(q_terms)))


class SearchForm(FlaskForm):
    search = StringField('')
    submit = SubmitField('Search')


@app.route("/<query>")
def return_links(query):
    q_terms = [term.lower() for term in query.strip().split()]
    return jsonify(lc_calc_docs_sorted_order(q_terms)[:20:])

@app.route("/", methods=['GET', 'POST'])
def home():
    form = SearchForm()
    results = {}
    if form.validate_on_submit():
        query = form.search.data
        q_terms = [term.lower() for term in query.strip().split()]
        results["leetcode"] = []
        results["leetcode"] += lc_calc_docs_sorted_order(q_terms)[:20:]
        results["codeforces"] = []
        results["codeforces"] += cf_calc_docs_sorted_order(q_terms)[:20:]
    return render_template('index.html', form=form, results=results)



