import numpy as np
import laspy
import rasterio
from rasterio.transform import from_origin
from sklearn.cluster import KMeans
import whitebox
import os
import tempfile
import shutil
from packaging import version

class LidarProcessor:
    def __init__(self):
        self.wbt = whitebox.WhiteboxTools()
        
        # Configuration du chemin vers l'exécutable WhiteboxTools
        if 'WBT_DIR' in os.environ:
            self.wbt.exe_path = os.path.join(os.environ['WBT_DIR'], 'whitebox_tools')
        
        # Configuration des répertoires de travail
        self.temp_dir = os.environ.get('TEMP_DIR', '/app/temp')
        self.data_dir = os.environ.get('DATA_DIR', '/app/data')
        
        # Création des répertoires s'ils n'existent pas
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Configuration du répertoire de travail WhiteboxTools
        self.wbt.work_dir = self.temp_dir
        
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
                z = cleaned_data.z
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
        # Création d'un sous-répertoire temporaire unique
        temp_subdir = tempfile.mkdtemp(dir=self.temp_dir)
        
        try:
            # Chemins des fichiers
            temp_las = os.path.join(temp_subdir, "temp.las")
            dsm_path = os.path.join(temp_subdir, "dsm.tif")
            dem_path = os.path.join(temp_subdir, "dem.tif")
            chm_path = os.path.join(temp_subdir, "chm.tif")
            
            # Sauvegarde du fichier LAS
            las_data.write(temp_las)
            
            # Génération du DSM
            self.wbt.lidar_digital_surface_model(
                i=temp_las,
                output=dsm_path,
                resolution=resolution
            )
            
            # Génération du DEM
            self.wbt.remove_off_terrain_objects(
                i=dsm_path,
                output=dem_path,
                filter_size=25,
                slope_threshold=15.0
            )
            
            # Génération du CHM
            self.wbt.subtract(
                input1=dsm_path,
                input2=dem_path,
                output=chm_path
            )
            
            # Lecture des modèles générés
            with rasterio.open(dsm_path) as src:
                dsm = src.read(1)
            with rasterio.open(dem_path) as src:
                dem = src.read(1)
            with rasterio.open(chm_path) as src:
                chm = src.read(1)
            
            return {
                'dsm': dsm,
                'dem': dem,
                'chm': chm
            }
            
        except Exception as e:
            raise Exception(f"Erreur lors de la génération des modèles : {str(e)}")
            
        finally:
            # Nettoyage du répertoire temporaire
            try:
                shutil.rmtree(temp_subdir)
            except:
                pass
    
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