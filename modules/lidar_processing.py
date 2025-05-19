import numpy as np
import laspy
import rasterio
from rasterio.transform import from_origin
from sklearn.cluster import KMeans
import whitebox
import os
import tempfile
import shutil
import time
from packaging import version
from scipy.ndimage import median_filter

class LidarProcessor:
    def __init__(self):
        self.wbt = whitebox.WhiteboxTools()
        
        # Configuration du chemin vers l'exécutable WhiteboxTools
        if 'WBT_DIR' in os.environ:
            wbt_path = os.path.join(os.environ['WBT_DIR'], 'whitebox_tools')
            if os.path.exists(wbt_path):
                self.wbt.exe_path = wbt_path
            else:
                raise Exception(f"L'exécutable WhiteboxTools n'existe pas au chemin: {wbt_path}")
        
        # Configuration des répertoires de travail
        self.temp_dir = os.environ.get('TEMP_DIR', '/app/temp')
        self.data_dir = os.environ.get('DATA_DIR', '/app/data')
        
        # Création des répertoires s'ils n'existent pas
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Configuration du répertoire de travail WhiteboxTools
        self.wbt.work_dir = self.temp_dir
        
        # Vérification que WhiteboxTools est fonctionnel
        self._check_wbt_installation()
        
    def _check_wbt_installation(self):
        """Vérifie que WhiteboxTools est correctement installé et fonctionnel."""
        try:
            # Récupère la version de WhiteboxTools pour vérifier qu'il fonctionne
            version_info = self.wbt.version()
            print(f"WhiteboxTools version: {version_info}")
        except Exception as e:
            raise Exception(f"WhiteboxTools n'est pas correctement installé ou configuré: {str(e)}")
        
    def preprocess(self, file_path):
        """Prétraite les données LiDAR."""
        try:
            # Lecture du fichier
            las = laspy.read(file_path)
            
            # Conversion vers la version 1.4 si nécessaire
            current_version = version.parse(str(las.header.version))
            target_version = version.parse("1.4")
            
            if current_version < target_version:
                las = self._convert_to_latest(las)
            
            return las
        except Exception as e:
            raise Exception(f"Erreur lors du prétraitement : {str(e)}")
    
    def get_basic_stats(self, las_data):
        """Calcule les statistiques de base des données."""
        try:
            stats = {
                "Nombre total de points": len(las_data.points),
                "Version du fichier": str(las_data.header.version),
                "Format de points": las_data.header.point_format.id,
                "Système de coordonnées": las_data.header.parse_crs(),
                "Emprise": {
                    "X": (float(las_data.header.mins[0]), float(las_data.header.maxs[0])),
                    "Y": (float(las_data.header.mins[1]), float(las_data.header.maxs[1])),
                    "Z": (float(las_data.header.mins[2]), float(las_data.header.maxs[2]))
                }
            }
            return stats
        except Exception as e:
            raise Exception(f"Erreur lors du calcul des statistiques : {str(e)}")
    
    def clean_data(self, las_data, min_height=None, max_height=None):
        """Nettoie les données en supprimant les valeurs aberrantes."""
        try:
            # Copie des données
            cleaned_data = las_data.copy()
            
            # Application des filtres de hauteur
            if min_height is not None or max_height is not None:
                z = np.array(cleaned_data.z)  # Conversion en array NumPy standard
                mask = np.ones(len(z), dtype=bool)
                
                if min_height is not None:
                    mask &= (z >= min_height)
                if max_height is not None:
                    mask &= (z <= max_height)
                    
                # Application du masque
                cleaned_data = cleaned_data[mask]
            
            return cleaned_data
        except Exception as e:
            raise Exception(f"Erreur lors du nettoyage des données : {str(e)}")
    
    def generate_models(self, las_data, resolution=1.0):
        """Génère les modèles numériques (DSM, DEM, CHM)."""
        # Limitation de la résolution pour éviter des modèles trop volumineux
        min_resolution = 0.5  # Définir une résolution minimale
        if resolution < min_resolution:
            print(f"Résolution limitée à {min_resolution}m pour éviter les modèles trop volumineux")
            resolution = min_resolution
            
        # Limiter également le nombre de points utilisés pour la génération des modèles
        max_points = 1000000  # 1 million de points maximum
        if len(las_data.points) > max_points:
            print(f"Sous-échantillonnage des données pour la génération des modèles ({max_points}/{len(las_data.points)} points)")
            indices = np.random.choice(len(las_data.points), max_points, replace=False)
            x = np.array(las_data.x[indices])
            y = np.array(las_data.y[indices])
            z = np.array(las_data.z[indices])
        else:
            x = np.array(las_data.x)
            y = np.array(las_data.y)
            z = np.array(las_data.z)
        
        # Création d'un sous-répertoire temporaire unique
        temp_subdir = tempfile.mkdtemp(dir=self.temp_dir)
        print(f"Dossier temporaire créé: {temp_subdir}")
        
        try:
            # Déterminer les limites de la grille
            x_min, x_max = np.min(x), np.max(x)
            y_min, y_max = np.min(y), np.max(y)
            
            # Limiter la taille de la grille
            max_grid_size = 500  # Taille maximale de la grille
            grid_width = min(max_grid_size, int((x_max - x_min) / resolution) + 1)
            grid_height = min(max_grid_size, int((y_max - y_min) / resolution) + 1)
            
            # Recalculer la résolution effective
            resolution_x = (x_max - x_min) / (grid_width - 1)
            resolution_y = (y_max - y_min) / (grid_height - 1)
            
            print(f"Création de grilles: {grid_height} x {grid_width}")
            print(f"Résolution effective: {resolution_x:.2f}m x {resolution_y:.2f}m")
            
            # Initialiser les matrices pour DSM et DEM
            dsm = np.full((grid_height, grid_width), -9999, dtype=np.float32)  # Valeur par défaut pour les points sans données
            dem = np.full((grid_height, grid_width), -9999, dtype=np.float32)  # Valeur par défaut pour les points sans données
            
            # Remplir le DSM (utiliser le point le plus haut dans chaque cellule)
            for i in range(len(x)):
                x_idx = min(int((x[i] - x_min) / resolution_x), grid_width - 1)
                y_idx = min(int((y[i] - y_min) / resolution_y), grid_height - 1)
                
                if dsm[y_idx, x_idx] < z[i] or dsm[y_idx, x_idx] == -9999:
                    dsm[y_idx, x_idx] = z[i]
            
            # Filtrer les valeurs par défaut dans le DSM
            dsm_mask = dsm != -9999
            if not np.any(dsm_mask):
                raise Exception("Aucune donnée valide dans le DSM")
            
            # Remplir les trous dans le DSM avec une interpolation simple
            # Remplacer les valeurs manquantes par la moyenne des voisins valides
            for _ in range(3):  # Répéter quelques fois pour combler plus de trous
                dsm_copy = np.copy(dsm)
                for y_idx in range(grid_height):
                    for x_idx in range(grid_width):
                        if dsm[y_idx, x_idx] == -9999:
                            # Définir les limites de la fenêtre
                            y_start = max(0, y_idx - 1)
                            y_end = min(grid_height, y_idx + 2)
                            x_start = max(0, x_idx - 1)
                            x_end = min(grid_width, x_idx + 2)
                            
                            # Extraire la fenêtre
                            window = dsm[y_start:y_end, x_start:x_end]
                            
                            # Calculer la moyenne des valeurs valides
                            valid_values = window[window != -9999]
                            if len(valid_values) > 0:
                                dsm_copy[y_idx, x_idx] = np.mean(valid_values)
                
                dsm = dsm_copy
            
            # Estimer les valeurs pour le DEM (on suppose que les points les plus bas sont au sol)
            # On crée un filtre de fenêtre glissante pour trouver le point le plus bas dans chaque fenêtre
            window_size = max(3, int(5 / resolution))  # Fenêtre de 5m ou au moins 3 cellules
            window_size = min(window_size, 15)  # Limiter la taille de la fenêtre pour éviter les calculs trop lourds
            
            # Copier le DSM comme point de départ pour le DEM
            dem = np.copy(dsm)
            
            # Pour chaque cellule avec des données
            for y_idx in range(grid_height):
                for x_idx in range(grid_width):
                    if dsm[y_idx, x_idx] == -9999:
                        continue
                    
                    # Définir les limites de la fenêtre
                    y_start = max(0, y_idx - window_size // 2)
                    y_end = min(grid_height, y_idx + window_size // 2 + 1)
                    x_start = max(0, x_idx - window_size // 2)
                    x_end = min(grid_width, x_idx + window_size // 2 + 1)
                    
                    # Extraire la fenêtre
                    window = dsm[y_start:y_end, x_start:x_end]
                    
                    # Trouver la valeur minimum (en ignorant les valeurs par défaut)
                    valid_values = window[window != -9999]
                    if len(valid_values) > 0:
                        dem[y_idx, x_idx] = np.min(valid_values)
            
            # Lissage du DEM avec un filtre médian pour supprimer les anomalies
            dem_valid = dem != -9999
            dem_copy = np.copy(dem)
            dem_copy[~dem_valid] = np.nan
            dem_smooth = median_filter(dem_copy, size=3)
            dem[dem_valid] = dem_smooth[dem_valid]
            
            # Calculer le CHM (Canopy Height Model)
            chm = np.zeros_like(dsm)
            valid_mask = (dsm != -9999) & (dem != -9999)
            chm[valid_mask] = dsm[valid_mask] - dem[valid_mask]
            
            # Appliquer un seuil minimum au CHM pour éviter les valeurs négatives ou trop petites
            chm[chm < 0] = 0
            
            # Remplacer les valeurs manquantes par zéro dans le CHM
            chm[~valid_mask] = 0
            
            print("Modèles numériques générés avec succès.")
            
            return {
                'dsm': dsm,
                'dem': dem,
                'chm': chm
            }
            
        except Exception as e:
            error_message = str(e)
            print(f"Erreur lors de la génération des modèles : {error_message}")
            raise Exception(f"Erreur lors de la génération des modèles : {error_message}")
            
        finally:
            # Nettoyage du répertoire temporaire
            try:
                shutil.rmtree(temp_subdir, ignore_errors=True)
            except Exception as e:
                print(f"Erreur lors du nettoyage du dossier temporaire: {str(e)}")
    
    def classify_vegetation(self, chm_data, n_classes=5):
        """Classifie la végétation basée sur le CHM."""
        try:
            # Masque pour exclure les zones sans végétation
            veg_mask = chm_data > 0.5
            veg_heights = chm_data[veg_mask]
            
            if len(veg_heights) == 0:
                return None
            
            # Classification par K-means
            kmeans = KMeans(n_clusters=n_classes, random_state=42)
            veg_heights_2d = veg_heights.reshape(-1, 1)
            classes = kmeans.fit_predict(veg_heights_2d)
            
            # Création de la carte de classification
            classification_map = np.zeros_like(chm_data, dtype=np.uint8)
            classification_map[veg_mask] = classes + 1
            
            # Définition des types de végétation
            centers = kmeans.cluster_centers_.flatten()
            sorted_indices = np.argsort(centers)
            
            vegetation_types = {
                0: "Pas de végétation (< 0.5m)"
            }
            
            for i, idx in enumerate(sorted_indices):
                height = centers[idx]
                class_num = idx + 1
                
                if height < 1:
                    veg_type = "Végétation basse"
                elif height < 5:
                    veg_type = "Arbustes"
                elif height < 15:
                    veg_type = "Arbres moyens"
                else:
                    veg_type = "Grands arbres"
                    
                vegetation_types[class_num] = f"{veg_type} (~{height:.1f}m)"
            
            return {
                'classification_map': classification_map,
                'vegetation_types': vegetation_types,
                'cluster_centers': centers
            }
        except Exception as e:
            raise Exception(f"Erreur lors de la classification : {str(e)}")
    
    def _convert_to_latest(self, las_data):
        """Convertit les données LiDAR vers la version la plus récente."""
        try:
            new_las = laspy.convert(las_data, point_format_id=6)
            return new_las
        except Exception as e:
            raise Exception(f"Erreur lors de la conversion : {str(e)}")
