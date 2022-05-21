################################################################################
# INPUT : Les matrices comprenant les signaux BOLD                             #
# OUTPUT : Les features et les targets nécessaires à la classification         #
# Auteur : Dardenne Charline                                                   #
# Modifié pour la dernière fois le 19 mai 2022.                                #
################################################################################

import numpy as np
import scipy.io
from scipy.stats import wasserstein_distance
from sklearn_extra.cluster import KMedoids

# Choix des variables
donnees = 0 # 0 ou 1 : Choix du type de données à utiliser 
# 0 : RGSF 
# 1 : RGS
N = 8 # Nombre de fenêtres prises dans une seule série temporelle
W = 140 # Tailles des fenêtres
nb_clusters = 100 # Nombres de clusters pour les K-médoïdes

### GENERATION DES FEATURES ###
deltat = 2.46
list_valp = []

# Boucle sur les 18 patients
for p in range(1,19):

    # Récupération des signaux BOLD
    if donnees == 0:
        mat = scipy.io.loadmat('../data_fMRI/Mat_data_reduced_GS_filtered_sub'+str(p)+'.mat')
    elif donnees == 1:
        mat = scipy.io.loadmat('../data_fMRI/Mat_data_reduced_GS_sub'+str(p)+'.mat')
    mat = mat['Mat_data_reduced'][0]
    
    # Boucle sur les états
    for e in range(4):

        # Sélection de l'état de conscience
        data = mat[e].transpose()

        # Détermination du nombre de colonnes
        m = len(data[0])

        # Génération du vecteur contenant les index auxquels extraires les fenêtres
        index = np.random.randint(1,m-W,N-2).tolist()
        index.append(m-W)
        index.insert(0,0)

        # Data augmentation (Window Slicing)
        for element in index:
             
            # Extraction d'une fenêtre
            sub_data = data[:,element:element+W]
            # Détermination du nombre de colonnes de la fenêtre
            sub_m = len(sub_data[0])

            # DMD exacte
            X = sub_data[:,0:sub_m-1] # m-1 premiers états
            Y = sub_data[:,1:sub_m] # m-1 derniers états

            U,s,Vh = np.linalg.svd(X, full_matrices=False)
            S = np.diag(s)
            V = np.asmatrix(Vh).getH()
            A = np.asmatrix(U).getH() @ Y @ V @ np.linalg.inv(S)

            # Extraction des valeurs propres et vecteurs propres
            valp, vecp = np.linalg.eig(A)
            list_valp.append(abs(valp))


# Caclul de la distance de Wassertein sur les vecteurs de valeurs propres
dist_wasserstein = np.empty((len(list_valp),len(list_valp)))
for i in range(len(list_valp)):
    for j in range(len(list_valp)):
        dist_wasserstein[i][j] = wasserstein_distance(list_valp[i],list_valp[j])

# K-médoïdes
kmedoids = KMedoids(n_clusters=nb_clusters, init='k-medoids++').fit(dist_wasserstein)

# Regroupement des fenêtres pour retrouver les séries temporelles d'origine
M = []
for i in range(0,len(kmedoids.labels_),N):
    M.append(kmedoids.labels_[i:i+N])

M = np.array(M)

# TF-IDF
# Calcul du terme TF
TF = np.zeros(((e+1)*p, nb_clusters), dtype=float)
for d in range((e+1)*p):
     for w in range(N):
         TF[d, M[d, w]] += 1

TF /= np.tile(np.sum(TF, 1).reshape((-1, 1)), (1, nb_clusters))

# Calcul du terme IDF
DF = np.zeros((nb_clusters,), dtype=float)
for d in range((e+1)*p):
     is_present = np.zeros((nb_clusters,), dtype=float)
     for w in range(N):
         is_present[M[d, w]] = 1

     DF += is_present

DF /= (e+1)*p

# Calcul de TF-IDF
TF_IDF = TF / np.tile(np.log(1/DF).reshape((1, -1)), ((e+1)*p, 1))

# Sauvegarde des features
if donnees == 0:
    np.save('features_RGSF.npy',TF_IDF)
elif donnees == 1:
    np.save('features_RGS.npy',TF_IDF)


### GENERATION DES TARGETS ###
targets = []

for i in range(18):
    targets.append(np.array([1,2,3,4]))

targets = np.array(targets).reshape(len(TF_IDF))

# Sauvegarde des targets
np.save('targets.npy',targets)