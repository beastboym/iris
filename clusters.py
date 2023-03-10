import numpy as np
import matplotlib.pyplot as plt

# On va ici utiliser des fonctions de la librairie NumPy
# (www.numpy.org) pour générer aléatoirement des données, autour d'un
# centre. Il s'agit de données générées à partir d'une distribution
# normale, soit d'une gaussienne. Les paramètres fournis à la fonction
# sont le centre, c'est à dire le point autour duquel les autres
# seront répartis, soit la moyenne de la gaussienne ; la covariance :
# la matrice diagonale d'étalement des éléments ; ainsi que le nombre
# de points à générer.
plt.ion()
centre1 = np.array([3, 3])  # centre du premier cluster
centre2 = np.array([-3, -3])  # centre du second cluster
sigma1 = np.array([[1.5, 0], [0, 1.5]])  # matrice de covariance du premier cluster
sigma2 = np.array([[1.5, 0], [0, 1.5]])  # matrice de covariance du second cluster
taille1 = 200  # nombre de points du premier cluster
taille2 = 200  # nombre de points du second cluster
cluster1 = np.random.multivariate_normal(centre1, sigma1, taille1)
cluster2 = np.random.multivariate_normal(centre2, sigma2, taille2)

# On utilise ici les fonctions de MatPlotLib (matplotlib.org) pour afficher les points.
# On commence par générer les listes d'abcisses, puis d'ordonnées pour chaque cluster.
# Ensuite, on ajouter les points aux clusters, rose pour le cluster1, bleu pour cluster2.
plt.scatter(
    [point[0] for point in cluster1], [point[1] for point in cluster1], color="pink"
)
plt.scatter(
    [point[0] for point in cluster2], [point[1] for point in cluster2], color="blue"
)
# plt.scatter(centre1[0], centre1[1], color="red") # coloration en rouge du centre du cluster1
# plt.scatter(centre2[0], centre2[1], color="red") # coloration en rouge du centre du cluster2
plt.show()
