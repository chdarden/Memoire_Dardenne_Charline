############################################################################################
# INPUT : Un fichier texte avec le choix du paramètre par la classe gridsearch             #
# OUTPUT : Le même fichier texte avec les fréquences d'apparition de chaque valeur         #
# Auteur : Dardenne Charline                                                               #
# Modifié pour la dernière fois le 19 mai 2022.                                            #
############################################################################################

# Ouverture du fichier texte
file = open('gs_param.txt', 'r+')

# Récupération des valeurs du paramètre choisies par la méthode grid search
data = []
for y in file.readlines():
    data.append(float(y))

# Calcul de l'occurence de chaque valeur 
list = [3,4,5,8,10,20,30]
results = []
for item in list:
    results.append(data.count(item))

# Calcul de la fréquence de chaque valeur 
results = [i/len(data) for i in results]

# Ecriture des fréquences dans le fichier texte
file.write("Fréquence : ")
for item in results :
        file.write('%f  ' % item)

file.close()