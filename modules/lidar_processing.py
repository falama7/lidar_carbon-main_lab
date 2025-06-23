import os
import whitebox
import laspy
from modules.utils import get_output_path, get_input_path

class LidarProcessor:
    def __init__(self):
        """
        Initialise le LidarProcessor.
        Configure le répertoire pour WhiteboxTools afin d'éviter le téléchargement automatique
        sur des systèmes de fichiers en lecture seule comme Streamlit Cloud.
        """
        # --- DÉBUT DE LA CORRECTION ---
        # Construit le chemin absolu vers le répertoire 'wbt' que vous avez ajouté à votre projet.
        # os.path.dirname(os.path.abspath(__file__)) -> obtient le chemin du dossier courant ('modules')
        # os.path.join(..., '..', 'wbt') -> remonte d'un niveau et va dans le dossier 'wbt'
        wbt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'wbt')

        # Initialise WhiteboxTools
        self.wbt = whitebox.WhiteboxTools()

        # Définit le répertoire de travail pour WhiteboxTools
        self.wbt.set_wbt_dir(wbt_dir)
        # --- FIN DE LA CORRECTION ---

        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_dsm(self, input_file, output_file):
        """
        Génère un Digital Surface Model (DSM) à partir d'un fichier Lidar.
        """
        input_path = get_input_path(input_file)
        output_path = get_output_path(self.output_dir, output_file)
        self.wbt.lidar_tin_gridding(
            i=input_path,
            output=output_path,
            interpolation_parameter='max',
            returns='all',
            resolution=1.0,
            tool_mode='tin'
        )
        return output_path

    def generate_dem(self, input_file, output_file):
        """
        Génère un Digital Elevation Model (DEM) à partir d'un fichier Lidar.
        Filtre les points au sol et interpole.
        """
        input_path = get_input_path(input_file)
        ground_points_file = get_output_path(self.output_dir, "ground_points.las")
        output_path = get_output_path(self.output_dir, output_file)

        # Classifie les points au sol
        self.wbt.classify_lidar_from_bare_earth(i=input_path, output=ground_points_file)

        # Crée le DEM à partir des points au sol
        self.wbt.lidar_tin_gridding(
            i=ground_points_file,
            output=output_path,
            interpolation_parameter='max',
            returns='ground',
            resolution=1.0
        )
        return output_path

    def generate_chm(self, dsm_file, dem_file, output_file):
        """
        Génère un Canopy Height Model (CHM) en soustrayant le DEM du DSM.
        """
        dsm_path = get_output_path(self.output_dir, dsm_file)
        dem_path = get_output_path(self.output_dir, dem_file)
        output_path = get_output_path(self.output_dir, output_file)
        self.wbt.subtract(input1=dsm_path, input2=dem_path, output=output_path)
        return output_path

    def segment_canopy(self, chm_file, output_file, min_height=2.0, max_crown_area=100.0):
        """
        Segmente la canopée à partir du CHM pour identifier les cimes d'arbres individuelles.
        """
        chm_path = get_output_path(self.output_dir, chm_file)
        output_path = get_output_path(self.output_dir, output_file)
        self.wbt.watershed_segmentation(
            d8_pntr=chm_path,  # Utilise le CHM comme entrée de direction de flux
            basins=output_path,
            pour_points=None,  # Pas de points de déversement spécifiques
            max_depth=None,
            max_size=max_crown_area
        )
        return output_path

    def extract_tree_info(self, lidar_file, chm_file, segmented_canopy_file):
        """
        Extrait des informations détaillées sur chaque arbre segmenté.
        (Cette fonction est un placeholder et nécessite une implémentation plus détaillée)
        """
        # Cette section nécessiterait une logique complexe pour :
        # 1. Lire le raster de la canopée segmentée.
        # 2. Pour chaque ID de segment (chaque arbre) :
        #    a. Extraire les points LiDAR qui tombent dans le polygone de ce segment.
        #    b. Calculer des métriques (hauteur max, diamètre de la couronne, etc.).
        # Pour la simplicité, nous retournons un message placeholder.
        print(f"Extraction d'infos pour {lidar_file}, {chm_file}, {segmented_canopy_file}")
        return {"tree_count": "Not Implemented", "average_height": "Not Implemented"}