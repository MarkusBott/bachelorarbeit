from itertools import compress, product
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


#svm classification 
def svm(data, target, tst_sz=0.2, rndm_stt=104, noisy=True):
    start = perf_counter()
    X_train, X_test, y_train, y_test = train_test_split(data,target,
                                           test_size=tst_sz,
                                           random_state=rndm_stt,
                                           shuffle=True)
    svc = SVC(kernel='sigmoid', gamma=0.1)
    model = svc.fit(X_train, y_train)
    middle = perf_counter()
    if(noisy):
        print(f"SVM model fitted in {middle-start} second(s)")
    predictions = model.predict(X_test)
    end = perf_counter()
    if(noisy):
        print(f"SVM predicted the testset in {end-middle} second(s)")
    report = classification_report(y_test, 
                                predictions,
                                output_dict=True,
                                zero_division=1)
    pd_report = pd.DataFrame(report)
    return pd_report, X_test, y_test, predictions

#use the results from k_means_cluster_search and classify the data with a svm 
#but we use this result instead of the original labels and then return the 
#best result when you try to reproduce the original labels
def svm_merging_hypothesis(cluster, data, target,tst_sz=0.2, rndm_stt=104):
    X_train, X_test, y_train, y_test = train_test_split(data,target,
                                           test_size=tst_sz,
                                           random_state=rndm_stt,
                                           shuffle=True)
    start = perf_counter()
    #search the best possible k-means clustering
    report_kmeans, temp_labels, kmeans_labels = search_best_for_n_cluster(cluster, 
                                                                X_train, 
                                                                y_train,
                                                                rndm_stt=rndm_stt,
                                                                random=True, 
                                                                noisy=False)
    #svm classification with the determined kmeans 
    # labels instead of the original labels
    svc = SVC(kernel='sigmoid', gamma=0.1)
    model = svc.fit(X_train, temp_labels)
    predictions = model.predict(X_test)
    #Try any possible cluster merge and return the best
    # -> we can assume that both labels are represented in the dataset. 
    #    Therfore is the case with one empty label and one with all 
    # determined labels is deleted
    report = classification_report(y_test, 
                        predictions,
                        output_dict=True,
                        zero_division=1)
    pd_report = pd.DataFrame(report)
    end = perf_counter()
    #Sting contains: It took {end-start} second(s) to 
    #complete the svm classification combined with k-means - merging hypothesis
    print(f'It took {end-start} second(s) to complete the svm classification combined with k-means - merging hypothesis')
    return pd_report, X_test, y_test, predictions
    
#create a list with the inertias for the elbow graph.
def elbow_inertias_list(data,c,random=False, n_nt=20, rndm_stt=104):
    inertias = []
    for i in range(1,c):
        if(random):
            kmeans = KMeans(
                n_clusters=i,
                init='random',
                n_init=n_nt,
                random_state=rndm_stt)
        else:
            kmeans = KMeans(
                n_clusters=i, 
                init="k-means++", 
                n_init=n_nt,
                random_state=rndm_stt)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    return inertias

#add the calculated labels to the dataset
def add_labels(X,labels):
    new_X = np.zeros((X.shape[0],X.shape[1]+1))
    new_X[0:X.shape[0],0:X.shape[1]] = X
    i = 0
    for row in new_X:
        new_X[i,(new_X.shape[1]-1)] = labels[i]
        i+=1
    return new_X

#cluster the dataset with k-means -> add the 'c' 
#different cluster as a new feature to the dataset 
# -> classification with svm
def k_means_cluster_search(
                        data,
                        target,
                        max_c,
                        noisy=False,
                        n_nt=40,
                        rndm_stt=104):
    pd_report = pd.DataFrame(
                    columns=['Number_of_clusters','Accuracy','F_one_Score','Time'])
    for c in range(2,(max_c+1)):
        start = perf_counter()
        if(noisy):
            print('Cluster:',c)
        kmeans = KMeans(
                    n_clusters=c,
                    init='random',
                    n_init=n_nt,
                    random_state=rndm_stt)
        kmeans.fit(data)
        new_data = add_labels(data,kmeans.labels_)
        svm_report, X_test, y_test, y_pred = svm.svm(new_data,target,noisy=False)
        end = perf_counter()
        time = end-start
        acc = svm_report["accuracy"][0]
        f_one = svm_report.get("0",{}).get("f1-score")
        pd_report.loc[len(pd_report)] = [c, acc, f_one,time]
    return pd_report

#same procedure as we do in 'k_means_cluster_search', 
#but instead of adding the k-means cluster, we add a labeling
#created by search_best_for_n_cluster
def k_means_cluster_search_merging_hypothesis(data, target, c):
    pd_report = pd.DataFrame(columns=['Accuracy','F_one_Score','Time'])
    start = perf_counter()
    kmeans_report, pred_labels, kmeans_labels = search_best_for_n_cluster(c,
                                                                data,
                                                                target)
    new_data = add_labels(data,pred_labels)
    svm_report, X_test, y_test, y_pred = svm.svm(new_data,target,noisy=False)
    end = perf_counter()
    time = end-start
    acc = svm_report["accuracy"][0]
    pd_report.loc[len(pd_report)] = [acc, 
                                     svm_report.get("0",{}).get("f1-score"),time]
    return pd_report

#create power set of items without {} and items. (output is sorted by length)
def combinations(items):
    res =  list(set(compress(items,mask)) for mask in product(*[[0,1]]*len(items)))
    res.pop(0)
    res.pop((len(res)-1))
    res.sort(key=len)
    return res

#search the best k-means performing the merge hypothesis
def search_best_for_n_cluster(
                                cluster,
                                data,
                                target,
                                breakloop=10,
                                noisy=False,
                                random=True,
                                n_nt=40,
                                rndm_stt=104):
    max_acc = 0
    max_f_one = 0
    labels = []
    final_report = {}
    start = perf_counter()
    for i in range(breakloop):
        if(noisy):
            print("iteration step ", i)
        if(random):
            kmeans = KMeans(
                    n_clusters=cluster,
                    init='random',
                    n_init=n_nt,
                    random_state=rndm_stt)
        else:
            kmeans = KMeans(
                    n_clusters=cluster,
                    init='k-means++',
                    n_init=n_nt,
                    random_state=rndm_stt)
        kmeans.fit(data)
        #Try any possible cluster merge and return the best
        # -> we can assume that both labels are represented in the dataset.
        #Therfore is the case with one empty label and one with all 
        #determined labels is deleted
        df = pd.DataFrame()
        df['label'] = kmeans.labels_
        combi_list = combinations(range(cluster))#power set of range(cluster)
        #without {} and {1,...,cluster} -> example: [1,2,3] 
        #-> [{3}, {2}, {1}, {2, 3}, {1, 3}, {1, 2}]
        rvs_combi_list = combi_list.copy()
        rvs_combi_list.reverse()#[{3}, {2}, {1}, {2, 3}, {1, 3}, {1, 2}]
        #-> [{1, 2}, {1, 3}, {2, 3}, {1}, {2}, {3}]
        #combi_list for label 0 and rvs_combi_list for label 1 create all
        #possible merging opportunities
        for i in range(len(combi_list)):
            temp_data = df.copy(deep=True)
            first_label = combi_list[i]
            second_label = rvs_combi_list[i]
            #map all clusters to the same label. 'x' and 'y' to avoid conflicts,
            #because when you do the same with ints it can produce crap.
            for j in first_label:
                temp_data.loc[temp_data['label']==j, 'label'] = 'x'
            for k in second_label:
                temp_data.loc[temp_data['label']==k,'label'] = 'y'
            #back to the original labels
            temp_data.loc[temp_data['label']=='x', 'label'] = 0
            temp_data.loc[temp_data['label']=='y', 'label'] = 1
            temp_labels = np.array(temp_data['label'],dtype=int)
            report = classification_report(target,temp_labels,output_dict=True)
            acc = report['accuracy']
            f_one = report.get("0",{}).get("f1-score")
            #store the best classification
            if(acc > max_acc and f_one > max_f_one):
                max_acc = acc
                max_f_one = f_one
                labels = temp_labels
                final_report = report
            del temp_data
    pd_report = pd.DataFrame(final_report)
    k_means_labels = kmeans.labels_
    end = perf_counter()
    #String contains: It took {end-start} second(s) to search
    #the best k-means clustering for merging hypothesis
    print(f'It took {end-start} second(s) to search the best k-means clustering for merging hypothesis')
    return pd_report, labels, k_means_labels