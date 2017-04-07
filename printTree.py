import pydotplus
from sklearn import tree


def display(clf, features):
    dData = tree.export_graphviz(clf, out_file=None,
                                 feature_names=features,
                                 class_names=['No Playoffs', 'Playoffs'])
    graph = pydotplus.graph_from_dot_data(dData)
    graph.write_pdf("out.pdf")
