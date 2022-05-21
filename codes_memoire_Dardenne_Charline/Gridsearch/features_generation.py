################################################################################
# INPUT : Les matrices comprenant les signaux BOLD                             #
# OUTPUT : Les features et les targets nécessaires à la classification         #
# Auteur : Dardenne Charline                                                   #
# Modifié pour la dernière fois le 19 mai 2022.                                #
################################################################################

import numpy as np
import scipy.io

### GENERATION DES FEATURES ###
deltat = 2.46
features = []

# Boucle sur les 18 patients
for p in range(1,19):
        
    # Récupération des signaux BOLD
    mat = scipy.io.loadmat('../data_fMRI/Mat_data_reduced_GS_sub'+str(p)+'.mat')
    mat = mat['Mat_data_reduced'][0]

    # Création d'une liste intermédiaire pour stocker les features de chaque patient
    vclass = []

    # Boucle sur les états
    for e in range(4):

        # Sélection de l'état de conscience
        data = mat[e].transpose()

        # Détermination du nombre de colonnes
        m = len(data[0])

        # DMD exacte
        X = data[:,0:m-1] # m-1 premiers états
        Y = data[:,1:m] # m-1 derniers états

        U,s,Vh = np.linalg.svd(X, full_matrices=False) 
        S = np.diag(s)
        V = np.asmatrix(Vh).getH()
        A = np.asmatrix(U).getH() @ Y @ V @ np.linalg.inv(S)

        # Extraction des valeurs propres et vecteurs propres
        valp, vecp = np.linalg.eig(A)

        # Calcul des valeurs propres en temps continu
        valp_cont = np.log(valp)/deltat

        # Calcul des modes
        modes = U @ vecp 

        # Tri selon la partie réelle des valeurs propres (croissant)
        real_valp = [x.real for x in valp_cont]
        sorted_valp_cont = [valp_cont[i] for i in np.argsort(real_valp)]
        sorted_modes = np.array([modes[:,i] for i in np.argsort(real_valp)])

        # Génération des features
        mdomc = abs(sorted_modes[:,len(sorted_modes)-1])
        valp = np.array([x.real for x in valp])
        mean_valp = np.mean(valp)
        vclass_etat = np.append(mdomc,mean_valp).reshape(1,len(sorted_modes)+1)
        vclass.append(vclass_etat)
    
    vclass = np.array(vclass).reshape(e+1,len(vclass_etat[0]))
    features.append(vclass) 

features = np.array(features).reshape((e+1)*p,len(vclass[0]))

# Sauvegarde des features
np.save('features_RGS.npy',features)


### GENERATION DES TARGETS ###
targets = []

for i in range(18):
    targets.append(np.array([1,2,3,4]))

targets = np.array(targets).reshape(len(features))

# Sauvegarde des targets
np.save('targets.npy',targets)