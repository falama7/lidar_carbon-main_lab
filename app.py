import streamlit as st
import os
import tempfile
import zipfile
from datetime import datetime
from modules.lidar_processing import LidarProcessor
from modules.visualization import DataVisualizer
from modules.carbon_estimation import CarbonEstimator
from modules.report_generator import ReportGenerator

os.environ["STREAMLIT_SERVER_MAX_MESSAGE_SIZE"] = "5500000"

# Configuration de la page avec des paramètres pour optimiser les performances
st.set_page_config(
    page_title="LiDAR Carbon Stock Analyzer",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com/votre-repo/lidar-carbon-analyzer",
        "Report a bug": "https://github.com/votre-repo/lidar-carbon-analyzer/issues",
        "About": "Application d'analyse LiDAR et d'estimation du stock de carbone"
    }
)

# Fonction pour limiter la taille d'un jeu de données LiDAR pour l'affichage
def limit_display_size(data, max_points=5500000):
    """Limite la taille des données à afficher"""
    import numpy as np
    if hasattr(data, 'points') and len(data.points) > max_points:
        indices = np.random.choice(len(data.points), max_points, replace=False)
        return data[indices]
    return data

class App:
    def __init__(self):
        self.lidar_processor = LidarProcessor()
        self.visualizer = DataVisualizer()
        self.carbon_estimator = CarbonEstimator()
        self.report_generator = ReportGenerator()
        
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        if 'results' not in st.session_state:
            st.session_state.results = None

    def main(self):
        st.title("🌳 Analyse LiDAR et Estimation du Stock de Carbone")
        
        # Sidebar pour la navigation
        with st.sidebar:
            st.header("Navigation")
            page = st.radio(
                "Choisissez une étape:",
                ["Upload & Prétraitement", 
                 "Visualisation & Nettoyage",
                 "Modèles Numériques",
                 "Classification Végétation",
                 "Estimation Carbone",
                 "Rapport & Export"]
            )

        if page == "Upload & Prétraitement":
            self.upload_and_preprocess_page()
        elif page == "Visualisation & Nettoyage":
            self.visualization_and_cleaning_page()
        elif page == "Modèles Numériques":
            self.numeric_models_page()
        elif page == "Classification Végétation":
            self.vegetation_classification_page()
        elif page == "Estimation Carbone":
            self.carbon_estimation_page()
        elif page == "Rapport & Export":
            self.report_and_export_page()

    def upload_and_preprocess_page(self):
        st.header("Upload et Prétraitement des Données")
        
        uploaded_file = st.file_uploader("Chargez votre fichier LiDAR (LAZ/LAS)", type=['laz', 'las'])
        
        if uploaded_file:
            with st.spinner("Prétraitement des données en cours..."):
                # Sauvegarde temporaire du fichier
                temp_dir = tempfile.mkdtemp()
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                
                # Prétraitement
                try:
                    processed_data = self.lidar_processor.preprocess(temp_path)
                    st.session_state.processed_data = processed_data
                    st.success("Prétraitement terminé avec succès!")
                    
                    # Affichage des statistiques basiques
                    st.subheader("Statistiques des données")
                    stats = self.lidar_processor.get_basic_stats(processed_data)
                    st.write(stats)
                except Exception as e:
                    st.error(f"Erreur lors du prétraitement: {str(e)}")

    def visualization_and_cleaning_page(self):
        st.header("Visualisation et Nettoyage des Données")
        
        if st.session_state.processed_data is None:
            st.warning("Veuillez d'abord charger et prétraiter vos données.")
            return
        
        # Options de visualisation
        viz_type = st.selectbox(
            "Type de visualisation",
            ["Nuage de points 3D", "Distribution des hauteurs", "Densité des points"]
        )
        
        # Paramètres de nettoyage
        st.subheader("Paramètres de nettoyage")
        min_height = st.slider("Hauteur minimum (m)", -10.0, 10.0, 0.0)
        max_height = st.slider("Hauteur maximum (m)", 0.0, 100.0, 50.0)
        
        if st.button("Nettoyer les données"):
            with st.spinner("Nettoyage en cours..."):
                cleaned_data = self.lidar_processor.clean_data(
                    st.session_state.processed_data,
                    min_height=min_height,
                    max_height=max_height
                )
                st.session_state.processed_data = cleaned_data
                st.success("Nettoyage terminé!")

        # Affichage de la visualisation
        self.visualizer.display(st.session_state.processed_data, viz_type)

    def numeric_models_page(self):
        st.header("Création des Modèles Numériques")
        
        if st.session_state.processed_data is None:
            st.warning("Veuillez d'abord charger et prétraiter vos données.")
            return
        
        # Paramètres pour la création des modèles
        resolution = st.slider("Résolution (m)", 0.1, 5.0, 1.0)
        
        if st.button("Générer les modèles"):
            with st.spinner("Génération des modèles en cours..."):
                models = self.lidar_processor.generate_models(
                    st.session_state.processed_data,
                    resolution=resolution
                )
                st.session_state.models = models
                st.success("Modèles générés avec succès!")
                
                # Affichage des modèles
                self.visualizer.display_models(models)

    def vegetation_classification_page(self):
        st.header("Classification de la Végétation")
        
        if not hasattr(st.session_state, 'models'):
            st.warning("Veuillez d'abord générer les modèles numériques.")
            return
        
        n_classes = st.slider("Nombre de classes de végétation", 3, 10, 5)
        
        if st.button("Classifier la végétation"):
            with st.spinner("Classification en cours..."):
                classification = self.lidar_processor.classify_vegetation(
                    st.session_state.models['chm'],
                    n_classes=n_classes
                )
                st.session_state.classification = classification
                st.success("Classification terminée!")
                
                # Affichage des résultats
                self.visualizer.display_classification(classification)

    def carbon_estimation_page(self):
        st.header("Estimation du Stock de Carbone")
        
        if not hasattr(st.session_state, 'classification'):
            st.warning("Veuillez d'abord effectuer la classification de la végétation.")
            return
        
        # Paramètres pour l'estimation
        method = st.selectbox(
            "Méthode d'estimation",
            ["Allométrique", "Machine Learning", "Hybride"]
        )
        
        if st.button("Estimer le stock de carbone"):
            with st.spinner("Estimation en cours..."):
                carbon_results = self.carbon_estimator.estimate(
                    st.session_state.classification,
                    method=method
                )
                st.session_state.carbon_results = carbon_results
                st.success("Estimation terminée!")
                
                # Affichage des résultats
                self.visualizer.display_carbon_results(carbon_results)

    def report_and_export_page(self):
        st.header("Génération du Rapport et Export")
        
        if not hasattr(st.session_state, 'carbon_results'):
            st.warning("Veuillez d'abord effectuer l'estimation du stock de carbone.")
            return
        
        # Options du rapport
        include_maps = st.checkbox("Inclure les cartes", value=True)
        include_stats = st.checkbox("Inclure les statistiques détaillées", value=True)
        include_methods = st.checkbox("Inclure la description des méthodes", value=True)
        
        if st.button("Générer le rapport"):
            with st.spinner("Génération du rapport en cours..."):
                # Création du dossier temporaire pour les résultats
                temp_dir = tempfile.mkdtemp()
                
                # Génération du rapport
                report_path = self.report_generator.generate(
                    st.session_state.carbon_results,
                    include_maps=include_maps,
                    include_stats=include_stats,
                    include_methods=include_methods,
                    output_dir=temp_dir
                )
                
                # Création du ZIP avec tous les résultats
                zip_path = os.path.join(temp_dir, "resultats_complets.zip")
                with zipfile.ZipFile(zip_path, 'w') as zipf:
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            if file != "resultats_complets.zip":
                                file_path = os.path.join(root, file)
                                arcname = os.path.relpath(file_path, temp_dir)
                                zipf.write(file_path, arcname)
                
                # Téléchargement du ZIP
                with open(zip_path, 'rb') as f:
                    st.download_button(
                        label="Télécharger tous les résultats (ZIP)",
                        data=f,
                        file_name=f"resultats_lidar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip"
                    )

if __name__ == "__main__":
    app = App()
    app.main() 