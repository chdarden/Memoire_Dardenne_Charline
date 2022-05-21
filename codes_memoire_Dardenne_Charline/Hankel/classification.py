############################################################################################
# INPUT : Une matrice des features et un vecteur de targets                                #
# OUTPUT : Des fichiers contenant les métriques mesurant la qualité de la classification   #
# Auteur : Dardenne Charline                                                               #
# Modifié pour la dernière fois le 19 mai 2022.                                            #
############################################################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, log_loss
from sklearn.metrics import ConfusionMatrixDisplay

# Choix d'une variable
donnees = 0 # 0 ou 1 : Choix du type de données à utiliser 
# 0 : RGSF 
# 1 : RGS

# Importation des features en fonction de "donnees" et ouverture de fichiers texte pour retenir les valeurs des métriques
if donnees == 0 :
    X = np.load('features_RGSF.npy')
    f1 = open ('accuracy_RGSF.txt','w')
    f2 = open ('f1_score_RGSF.txt','w')
    f3 = open ('logloss_RGSF.txt','w')
elif donnees == 1 :
    X = np.load('features_RGS.npy')
    f1 = open ('accuracy_RGS.txt','w')
    f2 = open ('f1_score_RGS.txt','w')
    f3 = open ('logloss_RGS.txt','w')

# Importation des targets
y = np.load('targets.npy')


### CLASSIFICATION ###

# Boucle pour faire 10 classifications
for t in range(10):

    # Modèle de classification
    clf = ExtraTreesClassifier()

    list_accuracy=[]
    list_f1score = []
    list_logloss = []
    list_y_pred = []

    # Cross-validation (17 patients pour l'entrainement et 1 pour le test)
    for i in range(0,len(X)-1,4):
        X_test = X[i:i+4]
        X_train = np.delete(X,[index for index in range (i,i+4)],axis=0)
        y_test = y[i:i+4]
        y_train = np.delete(y,[index for index in range (i,i+4)])

        # Classification
        clf.fit(X_train,y_train)
        y_pred = clf.predict(X_test)

        list_y_pred.extend(y_pred)

        # Calcul de l'accuracy
        list_accuracy.append(accuracy_score(y_test, y_pred))
        # Calcul du F1-Score 
        list_f1score.append(f1_score(y_test, y_pred, average=None).tolist())
        # Calcul du log loss
        y_pred_probs = clf.predict_proba(X_test)
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
    if donnees == 0:
        plt.savefig('test_'+str(t+1)+'_RGSF')
    elif donnees == 1:
        plt.savefig('test_'+str(t+1)+'_RGS')

    print("Classification n°%d : OK" %(t+1))

f1.close()
f2.close()
f3.close()

# Ouverture des fichiers textes contenant les valeurs des métriques pour calculer leur moyenne
if donnees == 0 :
    f1 = open('accuracy_RGSF.txt', 'r+')
    f2 = open('f1_score_RGSF.txt', 'r+')
    f3 = open('logloss_RGSF.txt', 'r+')
elif donnees == 1:
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
mean_logloss = []
for y in f3.readlines():
    data.append(float(y))
mean_logloss = np.mean(data)

f3.write('Mean : %f\n' % mean_logloss)

f1.close()
f2.close()
f3.close()