################################################################################################
# INPUT : Les matrices comprenant les signaux BOLD                                             #
# OUTPUT : Des fichiers contenant les métriques mesurant la qualité de la classification       #
# Auteur : Dardenne Charline                                                                   #
# Modifié pour la dernière fois le 19 mai 2022.                                                #
################################################################################################

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, log_loss
from sklearn.metrics import ConfusionMatrixDisplay


## GENERATION DES FEATURES 1ERE CLASSIFICATION : HANKEL-DMD ##
deltat = 2.46
features = []

# Choix d'une variable
mc = 10 # Nombre de colonnes de la matrice de Hankel

# Boucle sur les 18 patients
for p in range(1,19):

    # Récupération des signaux BOLD
    mat = scipy.io.loadmat('../data_fMRI/Mat_data_reduced_GS_filtered_sub'+str(p)+'.mat')
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


## GENERATION DES TARGETS (TRAIN) 1ERE CLASSIFICATION ##
y_train_classif1 = []

for i in range(17):
    y_train_classif1.append(np.array([1,2,2,1]))

y_train_classif1 = np.array(y_train_classif1).reshape(4*17)


## GENERATION DES TARGETS (TEST) 1ERE CLASSSIFICATION ##
y_classif1 = []

for i in range(18):
    y_classif1.append(np.array([1,2,2,1]))

y_classif1 = np.array(y_classif1).reshape(len(features))


## GENERATION DES TARGETS (TRAIN) 2EME CLASSIFICATION ##
y_train_classif2 = []

for i in range(17):
    y_train_classif2.append(np.array([2,3]))

y_train_classif2 = np.array(y_train_classif2).reshape(2*17)


## GENERATION DES TARGETS FINALES ##
targets_finales = []

for i in range(18):
    targets_finales.append(np.array([1,2,3,1]))

targets_finales = np.array(targets_finales).reshape(len(features))


### CLASSIFICATION ###

# Ouverture de fichiers texte pour retenir les valeurs des métriques
f1 = open ('accuracy.txt','w')
f2 = open ('f1_score.txt','w')

# Boucle pour faire 10 classifications
for t in range(10):

    ## 1ERE CLASSIFICATION ##

    # Modèle de classification
    clf = ExtraTreesClassifier()

    # Liste de prédictions de la 1ère classification
    list_y_pred_classif1 = []

    # Cross-validation (17 patients pour l'entrainement et 1 pour le test)
    for i in range(0,len(features)-1,4):
        X_test = features[i:i+4]
        X_train = np.delete(features,[index for index in range (i,i+4)],axis=0)

        # Classification
        clf.fit(X_train,y_train_classif1)
        y_pred = clf.predict(X_test)

        list_y_pred_classif1.extend(y_pred)

    print("1ere classif OK")


    ## 2EME CLASSIFICATION ##
    
    # Sélection des données encore à classifier (celles dont la target est égale à 2)
    index = [i for i in range(len(list_y_pred_classif1)) if list_y_pred_classif1[i] == 2]
    # Génération d'un vecteur donnant les patients auxquels appartiennent ces données
    patient = [(i//4)+1 for i in index]
    # Génération d'un vecteur donnant les états auxquels correspondent ces données
    etat = [i%4 for i in index]


    # Modèle de classification
    clf = ExtraTreesClassifier(max_depth = 4)

    # Liste de prédictions de la 2ème classification
    list_y_pred_classif2 = []


    # Boucle sur les 18 patients 
    for p in range(1,19):

        ## GENERATION DES FEATURES (TEST) 2EME CLASSIFICATION : DMD EXACTE ##
        X_test = []
        y_test_classif2 = []

        # Récupération des signaux BOLD
        mat = scipy.io.loadmat('../data_fMRI/Mat_data_reduced_GS_sub'+str(p)+'.mat')
        mat = mat['Mat_data_reduced'][0]

        # Génération d'un vecteur donnant les états du patient p qu'il faut encore passer dans la 2ème classification
        index_test = [i for i in range(len(patient)) if patient[i] == p]

        # Condition pour passer au patient suivant si aucun des états du patient p ne doit être classifié par la 2ème classification
        if len(index_test) == 0:
            print('Pas de test pour le patient n°%d' % p)
            continue

        # Boucle sur les états
        for j in index_test:

            # Sélection de l'état de conscience
            data = mat[etat[j]].transpose()

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
            X_test.append(vclass_etat)

        X_test = np.array(X_test).reshape(len(index_test),len(vclass_etat[0]))


        ## GENERATION DES FEATURES (TRAIN) 2EME CLASSIFICATION : DMD EXACTE ##

        # Création d'une liste dans laquelle on va stocker les features
        X_train = []

        # Elimination du patient qui sera considéré en test
        index_patient_train = np.delete(np.arange(1,19),(p-1))

        # Boucle sur les autres patients
        for q in index_patient_train:

            # Création d'une liste intermédiaire pour stocker les features de chaque patient
            vclass = []

            # Récupération des signaux BOLD
            mat = scipy.io.loadmat('../data_fMRI/Mat_data_reduced_GS_sub'+str(q)+'.mat')
            mat = mat['Mat_data_reduced'][0]

            # Boucle sur les états (ME et PE)
            for j in [1,2]:

                # Sélection de l'état de conscience
                data = mat[j].transpose()

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

            vclass = np.array(vclass).reshape(2,len(vclass_etat[0])) 
            X_train.append(vclass)

        X_train = np.array(X_train).reshape(2*17,len(vclass[0]))


        # Classification
        clf.fit(X_train,y_train_classif2)
        y_pred = clf.predict(X_test)

        list_y_pred_classif2.extend(y_pred)

    print("2eme classif OK")

    # Insertion des targets que la classification vient de prédire
    for i, r in enumerate(index):
        list_y_pred_classif1[r] = list_y_pred_classif2[i]
    

    # Calcul de l'accuracy 
    accuracy = accuracy_score(list_y_pred_classif1, targets_finales)

    # Caclul du F1-Score 
    f1score = f1_score(list_y_pred_classif1, targets_finales, average=None).tolist()

    # Ecriture des valeurs des métriques dans les fichiers texte
    f1.write('%f\n' % accuracy)

    for item in f1score :
        f2.write('%f  ' % item)
    f2.write('\n')

    # Matrice de confusion
    conf_mat = confusion_matrix(targets_finales, list_y_pred_classif1)
    ls = ['C+R', 'ME', 'PE']
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=ls)
    disp.plot()

    plt.savefig('test_'+str(t+1))

    print("Classification n°%d : OK" %(t+1))

f1.close()
f2.close()

# Ouverture des fichiers textes contenant les valeurs des métriques pour calculer leur moyenne
f1 = open('accuracy.txt', 'r+')
f2 = open('f1_score.txt', 'r+')

# Calcul de la moyenne des métriques et écriture de la moyenne dans les fichiers texte
# accuracy
data = []
for y in f1.readlines():
    data.append(float(y))
mean_accuracy = np.mean(data)

f1.write('Mean : %f\n' % mean_accuracy)

# F1-score
data = []
for y in f2.readlines():
    line = y.split("  ")[:3]
    data.append([float(i) for i in line])
mean_f1score = np.mean(data, axis = 0)

f2.write("Mean : ")
for item in mean_f1score :
        f2.write('%f  ' % item)


f1.close()
f2.close()