from clustering import *
from sklearn import metrics


cluster = clustering(120)
cluster.kmeans_cluster()

chi = metrics.calinski_harabaz_score(cluster.data, cluster.data_label)

print(chi)