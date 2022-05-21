################################################################################
# INPUT : Les matrices comprenant les signaux BOLD                             #
# OUTPUT : Les features et les targets nécessaires à la classification         #
# Auteur : Dardenne Charline                                                   #
# Modifié pour la dernière fois le 19 mai 2022.                                #
################################################################################

import numpy as np
import scipy.io

# Choix des variables
donnees = 0 # 0 ou 1 : Choix du type de données à utiliser 
# 0 : RGSF 
# 1 : RGS
mc = 10 # Nombre de colonnes de la matrice de Hankel


### GENERATION DES FEATURES ###
features = []

# Boucle sur les 18 patients
for p in range(1,19):

    # Récupération des signaux BOLD
    if donnees == 0:
        mat = scipy.io.loadmat('../data_fMRI/Mat_data_reduced_GS_filtered_sub'+str(p)+'.mat')
    elif donnees == 1:
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

        # Calcul de mr
        mr = m+1-mc

        # Génération de la matrice de Hankel
        H = data[:,0:mc]
        for i in range(1,mr):
            H = np.concatenate((H,data[:,i:mc+i]), axis=0)

        # Hankel-DMD
        ck = np.linalg.pinv(H[:,:-1]) @ H[:,-1] 
        C = np.concatenate((np.eye(mc-1,mc-2,k=-1),ck.reshape(mc-1,1)), axis=1)

        # Extraction des valeurs propres et vecteurs propres
        valp, vecp = np.linalg.eig(C)

        # Génération des features
        vclass_etat = abs(valp)
        vclass.append(vclass_etat)

    vclass = np.array(vclass) 
    features.append(vclass)

features = np.array(features).reshape((e+1)*p,len(valp))

# Sauvegarde des features
if donnees == 0:
    np.save('features_RGSF.npy',features)
elif donnees == 1:
    np.save('features_RGS.npy',features)


### GENERATION DES TARGETS ###
targets = []

for i in range(18):
    targets.append(np.array([1,2,3,4]))

targets = np.array(targets).reshape(len(features))

# Sauvegarde des targets
np.save('targets.npy',targets)


