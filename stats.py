import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.feature_extraction.text as sk
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn import svm
from sklearn.cluster import AgglomerativeClustering, AffinityPropagation
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import LabelEncoder


# {'BWT': Bioweapons, 'SCT': Secterian Violence, 'ISG': Iranian SPecial Group, 'RIC': Rashid IED CEll, 'SUN': Sunni criminal, 'BRT': Baathist cell, 'SUN/ISG': 6}


def svmz():
    classy = svm.SVC(kernel='linear')
    x, x_train, x_test, y_train, y_test = load_dataset()
    classy.fit(x_train, y_train)
    pred = classy.predict(x_test)

    conf_mat = confusion_matrix(y_test, pred)
    mate = {'BRT': 0, 'BWT': 1, 'SUN': 5, 'ISG': 2, 'RIC': 3, 'SUN/ISG': 6, 'SCT': 4}
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=mate, yticklabels=mate)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title("svm")
    plt.show()


def logistic_regre():
    loggie = LogisticRegression()
    X, X_train, X_test, y_train, y_test = load_dataset()
    loggie.fit(X_train, y_train)
    prediction = loggie.predict(X_test)
    conf_mat = confusion_matrix(y_test, prediction)
    mate = {'BRT': 0, 'BWT': 1, 'SUN': 5, 'ISG': 2, 'RIC': 3, 'SUN/ISG': 6, 'SCT': 4}
    sns.heatmap(conf_mat, annot=True, fmt='d',
                xticklabels=mate, yticklabels=mate)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title("logisitc")
    plt.show()

    acc = loggie.score(X_test, y_test)
    print(acc)


# k nearest
def kmean():
    file_loc = "insert_path_to_data"
    df = pd.read_excel(file_loc, sheet_name='Sheet1', na_values=['NA'], usecols=[4])
    df.fillna('NA', inplace=True)
    vec = sk.TfidfVectorizer(stop_words='english', sublinear_tf=True)

    testy = df['Unstructured Text'].tolist()

    x = vec.fit_transform(testy)
    model = KMeans(n_clusters=6, init='k-means++', max_iter=1400, n_init=12)
    kmodel = KMeans(n_clusters=6, init='k-means++', max_iter=1400, n_init=12)
    model.fit(x.toarray())
    model.fit_predict(x.toarray())

    Y = vec.transform([
        "A man approached an American patrol with evidence regarding an IED factory.BCT forces detained a Sunni munitions trafficker after search of his car netted IED trigger devices. "])
    prediction = model.predict(Y)
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vec.get_feature_names()
    for i in range(6):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :len(order_centroids)]:
            print(' %s' % terms[ind]),
    print(terms)
    print(prediction, "asd")

    tsne_init = 'pca'  # could also be 'random'
    tsne_early_exaggeration = 20.0
    random_state = 1
    model = TSNE(n_components=2, random_state=random_state, init=tsne_init, perplexity=33,
                 early_exaggeration=tsne_early_exaggeration, n_iter=1000)

    colors = np.array([x for x in 'brgcmy'])

    tmp = model.fit_transform(x.toarray())

    kmodel.fit(tmp)

    plt.scatter(tmp[:, 0], tmp[:, 1], marker="x", c=colors[kmodel.labels_])

    plt.scatter(kmodel.cluster_centers_[:, 0], kmodel.cluster_centers_[:, 1], marker="x", s=100, c='k')
    plt.show()
    linkage_mat = linkage(tmp, 'ward')
    plt.figure(figsize=(7.5, 5))

    augmented_dendrogram(linkage_mat, truncate_mode='lastp',  # show only the last p merged clusters
                         p=30,  # show only the last p merged clusters
                         leaf_rotation=90.,
                         leaf_font_size=12.,
                         show_contracted=True,  # to get a distribution impression in truncated branches
                         )
    plt.show()


# https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

def agglo_clust():
    np.set_printoptions(threshold=sys.maxsize)
    store = list()
    agglom = AgglomerativeClustering(n_clusters=6, linkage="ward")
    store.append(agglom)
    affine = AffinityPropagation(preference=-50)

    store.append(affine)
    x, x_train, x_test, y_train, y_test = load_dataset()
    for i, algo in enumerate(store):
        tsne_init = 'random'  # could also be 'random' or pca
        tsne_early_exaggeration = 20.0
        random_state = 1
        model = TSNE(n_components=2, random_state=random_state, init=tsne_init, perplexity=30,
                     early_exaggeration=tsne_early_exaggeration, n_iter=1400)
        tsne = model.fit_transform(x_train, y_train)
        algo.fit_predict(tsne)
        label = algo.labels_
        if 0 < i:
            print(algo.cluster_centers_indices_)
        plt.scatter(tsne[:, 0], tsne[:, 1], marker="x", c=label)
        plt.title(algo)
        plt.show()


def naive_baye():
    gb = GaussianNB()
    multi = MultinomialNB()
    _, x_train, x_test, y_train, y_test = load_dataset()
    gb.fit(x_train, y_train)
    multi.fit(x_train, y_train)
    pred = gb.predict(x_test)
    multi_pred = multi.predict(x_test)
    print(pred, multi_pred)


def load_dataset():
    '''
     Cleans and splits the  excel data into the training and testing sets for the model.
    :return:
    '''
    file_loc = "insert_path_to_data"
    df = pd.read_excel(file_loc, sheet_name='Sheet1', na_values=['NA'], usecols="B,E")
    df.fillna('NA', inplace=True)

    df['Thread ID'] = df['Thread ID'].str.replace('\d+', '')
    labeler = LabelEncoder()
    tmp = df['Thread ID']
    df['Thread ID'] = labeler.fit_transform(df['Thread ID'])
    print(dict(zip(tmp, df['Thread ID'])))
    vec = sk.TfidfVectorizer(stop_words='english', sublinear_tf=True)
    testy = df['Unstructured Text'].tolist()

    x = vec.fit_transform(testy)
    # return the labels , then the data
    X_train, X_test, y_train, y_test = train_test_split(x.toarray(), df['Thread ID'], test_size=0.3, random_state=10)
    return x, X_train, X_test, y_train, y_test


# shows the merging of clusters and the distance between them when merged.
def augmented_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


# kmean()
# agglo_clust()
# svmz()
naive_baye()
# logistic_regre()
