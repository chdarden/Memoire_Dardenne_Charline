############################################################################################
# INPUT : Une matrice des features et un vecteur de targets                                #
# OUTPUT : Des fichiers contenant les métriques mesurant la qualité de la classification   #
# Auteur : Dardenne Charline                                                               #
# Modifié pour la dernière fois le 19 mai 2022.                                            #
############################################################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, log_loss
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import ConfusionMatrixDisplay

# Choix d'une variable
N = 8 # Nombre de fenêtres prises dans une série temporelle

# Importation des features et ouverture de fichiers texte pour retenir les valeurs des métriques
X = np.load('features_RGS.npy')
f1 = open ('accuracy_RGS.txt','w')
f2 = open ('f1_score_RGS.txt','w')
f3 = open ('logloss_RGS.txt','w')

# Génération des targets pour la matrice de confusion
y = np.array([1,2,3,4])
for i in range(17):
    y = np.concatenate((y,np.array([1,2,3,4])), axis=0)

# Génération des ensembles de train et de test pour les targets
y_train = np.array([[1]*N,[2]*N,[3]*N,[4]*N]).flatten()
for i in range(16):
    y_train = np.concatenate((y_train,np.array([[1]*N,[2]*N,[3]*N,[4]*N]).flatten()), axis=0)

y_test = np.array([1,2,3,4])


### CLASSIFICATION ###

# Boucle pour faire 10 classifications
for t in range(10):

    # Modèle de classification
    clf = ExtraTreesClassifier()

    list_accuracy=[]
    list_y_pred = []
    list_f1score = []
    list_logloss = []

    # Cross-validation (17 patients pour l'entrainement et 1 pour le test)
    for i in range(0,len(X)-1,4*N):
        X_test = X[i:i+(4*N)]
        X_train = np.delete(X,[index for index in range (i,i+4*N)],axis=0)

        # Classification
        clf.fit(X_train,y_train)
        y_pred_augmentation = clf.predict(X_test)

        # Extraction de la target la plus prédite pour toutes les fenêtres d'une série temporelle
        y_pred = []
        for row in np.reshape(y_pred_augmentation,(4,N)):
            occ = np.array([np.count_nonzero(row==1),np.count_nonzero(row==2),np.count_nonzero(row==3),np.count_nonzero(row==4)]) 

            y_pred.append(np.argmax(occ)+1)

        list_y_pred.extend(y_pred)

        # Calcul de l'accuracy
        list_accuracy.append(accuracy_score(y_test, y_pred))
        # Calcul du F1-Score 
        list_f1score.append(f1_score(y_test, y_pred, average=None).tolist())
        # Calcul du log loss
        y_pred_probs_augmentation = clf.predict_proba(X_test)
        # Moyenne des log loss par état
        y_pred_probs = []
        for i in range(0,len(y_pred_probs_augmentation)-1,N):
            y_pred_probs.append(np.mean(y_pred_probs_augmentation[i:i+N], axis=0).tolist())
        
        list_logloss.append(log_loss(y_test, y_pred_probs))


    # Moyenne accuracy
    accuracy = np.mean(list_accuracy)
    # Moyenne F1-score
    f1score = np.mean(list_f1score, axis=0)
    # Moyenne log loss
    logloss = np.mean(list_logloss)

    # Ecriture des valeurs des métriques dans les fichiers texte
    f1.write('%f\n' % accuracy)

    for item in f1score :
        f2.write('%f  ' % item)
    f2.write('\n')
    
    f3.write('%f\n' % logloss)

    # Matrice de confusion
    conf_mat = confusion_matrix(y, list_y_pred)
    ls = ['C', 'ME', 'PE','R']
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=ls)
    disp.plot()

    # Sauvegarde de la matrice de confusion
    plt.savefig('test_'+str(t+1)+'_RGS')

    print("Classification n°%d : OK" %(t+1))

f1.close()
f2.close()
f3.close()

# Ouverture des fichiers textes contenant les valeurs des métriques pour calculer leur moyenne
f1 = open('accuracy_RGS.txt', 'r+')
f2 = open('f1_score_RGS.txt', 'r+')
f3 = open('logloss_RGS.txt', 'r+')

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
    line = y.split("  ")[:4]
    data.append([float(i) for i in line])
mean_f1score = np.mean(data, axis = 0)

f2.write("Mean : ")
for item in mean_f1score :
        f2.write('%f  ' % item)

# log loss
data = []
for y in f3.readlines():
    data.append(float(y))
mean_logloss = np.mean(data)

f3.write('Mean : %f\n' % mean_logloss)

f1.close()
f2.close()
f3.close()