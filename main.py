from sklearn import cluster, datasets, metrics
from sklearn.model_selection import train_test_split
from models import myKmeans, myDBSCAN

def main():
    # Load iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    n_clusters = len(set(iris.target))
    print("number of cluster:", n_clusters)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print("Training data size: %d" % len(X_train))
    print("Testing data size: %d" % len(X_test))

    # KMeans algorithm
    test_models = {}
    test_models['Kmeans'] = cluster.KMeans(n_clusters=n_clusters)
    test_models['Spectral clustering'] = cluster.SpectralClustering(n_clusters=n_clusters)
    test_models['Affinity propagation'] = cluster.AffinityPropagation(random_state=0)
    test_models['Mean shift'] = cluster.MeanShift()
    test_models['DBSCAN'] = cluster.DBSCAN(eps=0.75, min_samples=5)
    test_models['My Kmeans'] = myKmeans(n_clusters=n_clusters)
    test_models['My DBSCAN'] = myDBSCAN(eps=0.75, min_samples=5)

    for n in test_models:
        print("="*20)
        print("Model: ", n)
        model_evaluate(X_train, X_test, y_train, y_test, test_models[n])

def model_evaluate(X_train, X_test, y_train, y_test, model, data_split_rate = 0.2):
    # Predict
    model.fit(X_train)    
    y_pred_val = model.labels_
    internal_score = metrics.rand_score(y_train, y_pred_val)
    print("Internal Prediction Score: %f" % internal_score)

    # Print prediction result
    if hasattr(model, 'predict'):
        y_pred_test = model.predict(X_test)
    elif hasattr(model, 'fit_predict'):
        y_pred_test = model.fit_predict(X_test)
    else:
        print("No External Prediction")
        return
    external_score = metrics.rand_score(y_test, y_pred_test)
    print("External Prediction Score: %f" % external_score)
    

if __name__ == '__main__':
    main()