# -*- coding: utf-8 -*-
"""
@author: Gonçalo Mateus, 53052
@author: Lourenço Vasconcelos, 52699
"""

#importing libraries
import numpy as np
from skimage.io import imread
from sklearn.feature_selection import f_classif
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.neighbors import NearestNeighbors 
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

"""----------------------------------------------------"""
#  Auxiliary functions
"""----------------------------------------------------"""
def images_as_matrix(N=563):
    """
    Reads all N images in the images folder (indexed 0 through N-1)
    returns a 2D numpy array with one image per row and one pixel per column
    """
    return np.array([imread(f'images/{ix}.png',as_gray=True).ravel() for ix in range(563)])
        
def report_clusters(ids, labels, report_file):
    """Generates html with cluster report
    ids is a 1D array with the id numbers of the images in the images/ folder
    labels is a 1D array with the corresponding cluster labels
    """
    diff_lbls = list(np.unique(labels))
    diff_lbls.sort()
    html = ["""<!DOCTYPE html>
    <html lang="en">
       <head>
        <meta http-equiv="content-type" content="text/html; charset=utf-8">
        <meta charset="UTF-8">
        <title>Cluster Report</title>
       </head>
       <body>
       """]
    for lbl in diff_lbls:
        html.append(f"<h1>Cluster {lbl}</h1>")        
        lbl_imgs = ids[labels==lbl]          
        for count,img in enumerate(lbl_imgs):                
            html.append(f'<img src="images/{int(img)}.png" />')
            #if count % 10 == 9:
            #    html.append('<br/>')
    html.append("</body></html>")   
    with open(report_file,'w') as ofil:
        ofil.write('\n'.join(html))


"""----------------------------------------------------"""
#  Preprocess: Load data
"""----------------------------------------------------"""

y = np.loadtxt('labels.txt', delimiter=',')
y2 = y[y[:,1]>0,:]
y_ix = y2[:,0]
y_labels=y2[:,1]
X = images_as_matrix()

# Extraction
pca = PCA(n_components=6)
pca.fit(X)
pca_data = pca.transform(X)

tsne = TSNE(n_components=6, method='exact')
tsne_data = tsne.fit_transform(X)

isom = Isomap(n_components=6)
iso_data = isom.fit_transform(X)

# All features togethers
matrix = np.concatenate((pca_data, tsne_data, iso_data), axis=1)     
matrix_f = np.concatenate((pca_data[y_ix.astype(int)], tsne_data[y_ix.astype(int)], iso_data[y_ix.astype(int)]), axis=1)
                
"""----------------------------------------------------"""
#  Features selection
"""----------------------------------------------------"""        

# Select features with correlation matrix using Pearson Correlation
df = pd.DataFrame(matrix, columns = np.arange(0,18,1))
plt.figure(figsize=(12,10))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Oranges)
plt.savefig('Heatmap.png', dpi=300)
plt.show()
plt.close()

threshold = 0.5
mostCorrelated = []
for i in range(0,17,1):
    for j in range(i+1,18,1):
        if(cor[i][j]>threshold): 
            if j not in mostCorrelated:
                mostCorrelated.append(j)

# Most Correlated features = [0, 12], [1, 13], [2, 13], [2, 14], [3, 15], [4, 16]
dataset=matrix
f, prob = f_classif(dataset[y[:,1]>0,:],y_labels)
print(f)
print(prob)
df = pd.DataFrame({'Features': np.arange(0,18,1), 'f-value': f})
ax = df.plot.barh(x='Features', y='f-value')
plt.title("18 Feature importance using F-test")
plt.show()
plt.close()

#Delete correlated features
dataset = np.delete(matrix, [0,3,13,14,16], 1)

# Select features with (ANOVA) F-test
f, prob = f_classif(dataset[y[:,1]>0,:],y_labels)
print(f)
print(prob)
df = pd.DataFrame({'Features': np.arange(0,13,1), 'f-value': f})
ax = df.plot.barh(x='Features', y='f-value')
plt.title("Feature importance using F-test")
plt.savefig('Feature_importance_Ftest.png', dpi=300)
plt.show()
plt.close()

dataset = dataset[:, [0,1,10]]

"""----------------------------------------------------"""
#  Standardize data
"""----------------------------------------------------"""

means = np.mean(dataset,axis=0)
stdevs = np.std(dataset,axis=0)
dataset = (dataset-means)/stdevs

"""----------------------------------------------------"""
#  DBSCAN: Determining the Parameter Eps
"""----------------------------------------------------"""

# Find the best eps parameter
nbrs = NearestNeighbors(n_neighbors=5).fit(dataset)
distances, indices = nbrs.kneighbors(dataset)
distanceDec = sorted(distances[:,5-1], reverse=False)
plt.figure(figsize=(12,10))
plt.title("Determining the Parameter Eps")
plt.plot(indices[:,0], distanceDec)
plt.savefig('Parameter_Eps.png', dpi=300)
plt.show()
plt.close()

# DBSCAN with the best eps parameter
dbscan = DBSCAN(eps=0.4, min_samples=5).fit(dataset)
report_clusters(y[:,0], dbscan.labels_, 'dbscan_eps_0.4.html')

dbscan = DBSCAN(eps=0.45, min_samples=5).fit(dataset)
report_clusters(y[:,0], dbscan.labels_, 'dbscan_eps_0.45.html')

dbscan = DBSCAN(eps=0.5, min_samples=5).fit(dataset)
report_clusters(y[:,0], dbscan.labels_, 'dbscan_eps_0.5.html')

dbscan = DBSCAN(eps=0.55, min_samples=5).fit(dataset)
report_clusters(y[:,0], dbscan.labels_, 'dbscan_eps_0.55.html')

dbscan = DBSCAN(eps=0.6, min_samples=5).fit(dataset)
report_clusters(y[:,0], dbscan.labels_, 'dbscan_eps_0.6.html')

"""----------------------------------------------------"""
#  Kmeans: Determining the Parameter k
"""----------------------------------------------------"""
kmeans = KMeans(n_clusters=2).fit(dataset)
report_clusters(y[:,0], kmeans.labels_, 'kmeans_k_2.html')

kmeans = KMeans(n_clusters=3).fit(dataset)
report_clusters(y[:,0], kmeans.labels_, 'kmeans_k_3.html')

kmeans = KMeans(n_clusters=4).fit(dataset)
report_clusters(y[:,0], kmeans.labels_, 'kmeans_k_4.html')

kmeans = KMeans(n_clusters=8).fit(dataset)
report_clusters(y[:,0], kmeans.labels_, 'kmeans_k_8.html')

kmeans = KMeans(n_clusters=13).fit(dataset)
report_clusters(y[:,0], kmeans.labels_, 'kmeans_k_13.html')

kmeans = KMeans(n_clusters=5).fit(dataset)
report_clusters(y[:,0], kmeans.labels_, 'kmeans_k_5.html')

kmeans = KMeans(n_clusters=11).fit(dataset)
report_clusters(y[:,0], kmeans.labels_, 'kmeans_k_11.html')
  
"""----------------------------------------------------"""
#  Kmeans: Performance varying the parameter k in range(2,21)
"""----------------------------------------------------"""
precision=[]
recall=[]
rand=[]
f1=[]
adj_rand=[]
silhouette=[]

rangeKm = np.arange(2,21,1);
for l in rangeKm:
    tp=0
    fp=0
    tn=0
    fn=0
    kmeans = KMeans(n_clusters=l).fit(dataset)
    labels1 = np.zeros(81)
    for i in range(0,81):
        labels1[i] = kmeans.labels_[int(y_ix[i])]
    
    for i in range(0,80):
        for j in range(i+1,81):
            if( y_labels[i]==y_labels[j] and labels1[i] == labels1[j]):
                tp=tp+1
            if(y_labels[i]!=y_labels[j] and labels1[i]!=labels1[j]):
                tn=tn+1
            if(y_labels[i]!=y_labels[j] and labels1[i]==labels1[j]):
                fp=fp+1
            if(y_labels[i]==y_labels[j] and labels1[i]!=labels1[j]):
                fn=fn+1
    precision_c = tp/(tp+fp)
    recall_c = tp/(tp+fn)
    precision.append(tp/(tp+fp))
    recall.append(tp/(tp+fn))
    rand.append((tp+tn)/((81*80)/2))
    f1.append(2*((precision_c*recall_c)/(precision_c+recall_c)))
    adj_rand.append(adjusted_rand_score(y_labels,labels1))
    silhouette.append(silhouette_score(dataset,kmeans.labels_))
    
plt.title("KMEANS indicators")
plt.plot(rangeKm, precision, "-",color="blue", label="precision")
plt.plot(rangeKm, recall, "-",color="red", label="recall")
plt.plot(rangeKm, rand, "-",color="green", label="rand")
plt.plot(rangeKm, f1, "-",color="black", label="f1")
plt.plot(rangeKm, adj_rand, "-",color="brown", label="adj_rand")
plt.plot(rangeKm, silhouette, "-",color="orange", label="silhouette")
plt.legend(loc=5)
plt.savefig('KMEANS_indicators.png', dpi=300)
plt.show()
plt.close()

"""----------------------------------------------------"""
#  DBSCAN: Performance varying the parameter eps in range(250,2000,10)
"""----------------------------------------------------"""
precision=[]
recall=[]
rand=[]
f1=[]
adj_rand=[]
silhouette=[]

rangeDb = np.arange(0.25,1.2,0.01);
for l in rangeDb:
    tp=0
    fp=0
    tn=0
    fn=0
    dbscan = DBSCAN(eps=l, min_samples=5).fit(dataset)
    labels1 = np.zeros(81)
    for i in range(0,81):
        labels1[i] = dbscan.labels_[int(y_ix[i])]
    
    for i in range(0,80):
        for j in range(i+1,81):
            if( y_labels[i]==y_labels[j] and labels1[i] == labels1[j] and labels1[i] != -1):
                tp=tp+1
            if(y_labels[i]!=y_labels[j] and labels1[i]!=labels1[j]):
                tn=tn+1
            if(y_labels[i]!=y_labels[j] and labels1[i]==labels1[j]  and labels1[i] != -1):
                fp=fp+1
            if(y_labels[i]==y_labels[j] and labels1[i]!=labels1[j]):
                fn=fn+1
    if ((tp+fp)==0):
        precision_c = 0        
    else:
        precision_c = tp/(tp+fp)
   
    if ((tp+fn)==0):
        recall_c = 0
    else:
        recall_c = tp/(tp+fn)

    if ((precision_c+recall_c)==0):
        f1.append(0)      
    else:
        f1.append(2*((precision_c*recall_c)/(precision_c+recall_c)))
        
    if (np.unique(dbscan.labels_).size>1):
        silhouette.append(silhouette_score(dataset,dbscan.labels_))
    else:
        silhouette.append(0)        
    
    precision.append(precision_c)
    recall.append(recall_c)
    rand.append((tp+tn)/((81*80)/2))
    adj_rand.append(adjusted_rand_score(y_labels,labels1))

plt.title("DBSCAN indicators")
plt.plot(rangeDb, precision, "-",color="blue", label="precision")
plt.plot(rangeDb, recall, "-",color="red", label="recall")
plt.plot(rangeDb, rand, "-",color="green", label="rand")
plt.plot(rangeDb, f1, "-",color="black", label="f1")
plt.plot(rangeDb, adj_rand, "-",color="brown", label="adj_rand")
plt.plot(rangeDb, silhouette, "-",color="orange", label="silhouette")
plt.legend(loc=5)
plt.savefig('DBSCAN_indicators.png', dpi=300)
plt.show()
plt.close()

"""----------------------------------------------------"""
#  Gaussian Mixture: Performance varying the parameter n_components in range(2,21,1)
"""----------------------------------------------------"""
from sklearn.mixture import GaussianMixture

precision=[]
recall=[]
rand=[]
f1=[]
adj_rand=[]
silhouette=[]

rangeGm = np.arange(2,21,1);
for l in rangeGm:
    tp=0
    fp=0
    tn=0
    fn=0
    gm = GaussianMixture(n_components=l).fit_predict(dataset)
    labels1 = np.zeros(81)
    for i in range(0,81):
        labels1[i] = gm[int(y_ix[i])]
    
    for i in range(0,80):
        for j in range(i+1,81):
            if( y_labels[i]==y_labels[j] and labels1[i] == labels1[j] and labels1[i] != -1):
                tp=tp+1
            if(y_labels[i]!=y_labels[j] and labels1[i]!=labels1[j]):
                tn=tn+1
            if(y_labels[i]!=y_labels[j] and labels1[i]==labels1[j]  and labels1[i] != -1):
                fp=fp+1
            if(y_labels[i]==y_labels[j] and labels1[i]!=labels1[j]):
                fn=fn+1
    if ((tp+fp)==0):
        precision_c = 0        
    else:
        precision_c = tp/(tp+fp)
   
    if ((tp+fn)==0):
        recall_c = 0
    else:
        recall_c = tp/(tp+fn)

    if ((precision_c+recall_c)==0):
        f1.append(0)      
    else:
        f1.append(2*((precision_c*recall_c)/(precision_c+recall_c)))
        
    if (np.unique(dbscan.labels_).size>1):
        silhouette.append(silhouette_score(dataset,dbscan.labels_))
    else:
        silhouette.append(0)        
    
    precision.append(precision_c)
    recall.append(recall_c)
    rand.append((tp+tn)/((81*80)/2))
    adj_rand.append(adjusted_rand_score(y_labels,labels1))

plt.title("GaussianMixture indicators")
plt.plot(rangeGm, precision, "-",color="blue", label="precision")
plt.plot(rangeGm, recall, "-",color="red", label="recall")
plt.plot(rangeGm, rand, "-",color="green", label="rand")
plt.plot(rangeGm, f1, "-",color="black", label="f1")
plt.plot(rangeGm, adj_rand, "-",color="brown", label="adj_rand")
plt.plot(rangeGm, silhouette, "-",color="orange", label="silhouette")
plt.legend(loc=5)
plt.savefig('GaussianMixture_indicators.png', dpi=300)
plt.show()
plt.close()


# Best GaussianMixture cluster
gm = GaussianMixture(n_components=5).fit_predict(dataset)
report_clusters(y[:,0], gm, 'GaussianMixture_comp_5.html')

# Best GaussianMixture cluster
gm = GaussianMixture(n_components=11).fit_predict(dataset)
report_clusters(y[:,0], gm, 'GaussianMixture_comp_11.html')

"""----------------------------------------------------"""
#  Bissecting K-Means hierarchical clustering algorithm
"""----------------------------------------------------"""

DIV_STYLE = """style = "display: block;border-style: solid; border-width: 5px;border-color:blue;padding:5px;margin:5px;" """
def cluster_div(prev,ids,lbl_lists):
    div = []    
    lbls = [lbl[0] for lbl in lbl_lists]
    lbls = list(np.unique(lbls))
    lbls.sort()
    for lbl in lbls:
        div.append(f'<div {DIV_STYLE}>\n<h1>Cluster{prev}{lbl}</h1>')        
        indexes = [ix for ix in range(len(ids)) if lbl_lists[ix][0]==lbl]
        current_indexes = [ix for ix in indexes if len(lbl_lists[ix]) == 1]
        next_indexes = [ix for ix in indexes if len(lbl_lists[ix]) > 1]
        for ix in current_indexes:
                div.append(f'<img src="images/{int(ids[ix])}.png" />')
        if len(next_indexes)>0:            
            #print(f'**{prev}**\n',indexes,'\n  ',current_indexes,'\n   ',next_indexes, len(next_indexes))        
            next_ids = [ids[ix] for ix in next_indexes]
            next_lbl_lists = [lbl_lists[ix][1:] for ix in next_indexes]
            #print('**',next_lbl_lists)
            div.append(cluster_div(f'{prev}{lbl}-',next_ids,next_lbl_lists))
        div.append('</div>')
    return '\n'.join(div)


def report_clusters_hierarchical(ixs,label_lists,report_file):
    html = ["""<!DOCTYPE html>
    <html lang="en">
       <head>
        <meta http-equiv="content-type" content="text/html; charset=utf-8">
        <meta charset="UTF-8">
        <title>Cluster Report</title>
       </head>
       <body>
       """]   
    html.append(cluster_div('',ixs,label_lists))   
    html.append("</body></html>")   
    with open(report_file,'w') as ofil:
        ofil.write('\n'.join(html))
        
    
kmeans = KMeans(n_clusters=1).fit(dataset)
labels_ = kmeans.labels_
for i in range(2,6):
    largest =[]
    largest_l=-1
    test=[]
    z=0
    for j in range(0,i):
        for k in kmeans.labels_:
            if(j==kmeans.labels_[k]):
                test.append(y[:,0][z])
                z+=1
        if(len(test)>len(largest)):
            largest = test
            largest_l= j
    test = np.array(test)
    largest_l = np.array(largest_l)
    new_dataset = []
    for j in largest:
        new_dataset.append(dataset[int(j)])
    new_dataset=np.array(new_dataset)
    new_kmeans = KMeans(n_clusters=i).fit(new_dataset)
    labels_ = [item+1 if item>largest_l else item for item in labels_]
    labels_ = [-1 if item == largest_l else item for item in labels_]
    labels_ = [item+2 if item<largest_l else item for item in labels_]
    labels_ = [item if item==-1 else item for item in new_kmeans.labels_]
    report_clusters(y[:,0], labels_, 'bissecting_kmeans_3.html')
    
    
kmeans = KMeans(n_clusters=1).fit(dataset)
labels_ = kmeans.labels_

for i in range(2,9):
    largest =[]
    largest_l=-1
    test=[]
    z=0
    for j in range(0,i):
        for k in kmeans.labels_:
            if(j==kmeans.labels_[k]):
                test.append(y[:,0][z])
                z+=1
        if(len(test)>len(largest)):
            largest = test
            largest_l= j
    test = np.array(test)
    largest_l = np.array(largest_l)
    new_dataset = []
    for j in largest:
        new_dataset.append(dataset[int(j)])
    new_dataset=np.array(new_dataset)
    new_kmeans = KMeans(n_clusters=i).fit(new_dataset)
    labels_ = [item+1 if item>largest_l else item for item in labels_]
    labels_ = [-1 if item == largest_l else item for item in labels_]
    labels_ = [item+2 if item<largest_l else item for item in labels_]
    labels_ = [item if item==-1 else item for item in new_kmeans.labels_]
    report_clusters(y[:,0], labels_, 'bissecting_kmeans_7.html')
    
    
kmeans = KMeans(n_clusters=1).fit(dataset)
labels_ = kmeans.labels_
for i in range(2,14):
    largest =[]
    largest_l=-1
    test=[]
    z=0
    for j in range(0,i):
        for k in kmeans.labels_:
            if(j==kmeans.labels_[k]):
                test.append(y[:,0][z])
                z+=1
        if(len(test)>len(largest)):
            largest = test
            largest_l= j
    test = np.array(test)
    largest_l = np.array(largest_l)
    new_dataset = []
    for j in largest:
        new_dataset.append(dataset[int(j)])
    new_dataset=np.array(new_dataset)
    new_kmeans = KMeans(n_clusters=i).fit(new_dataset)
    labels_ = [item+1 if item>largest_l else item for item in labels_]
    labels_ = [-1 if item == largest_l else item for item in labels_]
    labels_ = [item+2 if item<largest_l else item for item in labels_]
    labels_ = [item if item==-1 else item for item in new_kmeans.labels_]
    report_clusters(y[:,0], labels_, 'bissecting_kmeans_12.html')