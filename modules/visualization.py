import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class DataVisualizer:
    def __init__(self):
        self.color_maps = {
            'vegetation': ['#FFFFFF', '#C2DF23', '#1E6C0B', '#1A4314', '#0B2307'],
            'carbon': px.colors.sequential.Viridis
        }
    
    def display(self, las_data, viz_type):
        """Affiche les données LiDAR selon le type de visualisation choisi."""
        if viz_type == "Nuage de points 3D":
            self._display_point_cloud_3d(las_data)
        elif viz_type == "Distribution des hauteurs":
            self._display_height_distribution(las_data)
        elif viz_type == "Densité des points":
            self._display_point_density(las_data)
    
    def display_models(self, models):
        """Affiche les modèles numériques (DSM, DEM, CHM)."""
        # Sous-échantillonnage des modèles pour l'affichage
        max_size = 100  # Taille maximale pour l'affichage
        
        # Fonction pour sous-échantillonner les modèles
        def downsample_model(model):
            if model is None:
                return None
            h, w = model.shape
            h_step = max(1, h // max_size)
            w_step = max(1, w // max_size)
            return model[::h_step, ::w_step]
        
        # Sous-échantillonnage des modèles
        dsm_display = downsample_model(models.get('dsm'))
        dem_display = downsample_model(models.get('dem'))
        chm_display = downsample_model(models.get('chm'))
        
        # Création de 3 colonnes pour l'affichage
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("DSM")
            if dsm_display is not None:
                fig = self._create_surface_plot(dsm_display, "Modèle Numérique de Surface")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("DSM non disponible")
        
        with col2:
            st.subheader("DEM")
            if dem_display is not None:
                fig = self._create_surface_plot(dem_display, "Modèle Numérique de Terrain")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("DEM non disponible")
        
        with col3:
            st.subheader("CHM")
            if chm_display is not None:
                fig = self._create_surface_plot(chm_display, "Modèle de Hauteur de Canopée")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("CHM non disponible")
    
    def display_classification(self, classification_results):
        """Affiche les résultats de la classification de végétation."""
        if classification_results is None:
            st.warning("Pas de données de classification disponibles.")
            return
            
        # Sous-échantillonnage pour l'affichage si nécessaire
        max_size = 400  # Taille maximale pour l'affichage
        class_map = classification_results['classification_map']
        
        h, w = class_map.shape
        if h > max_size or w > max_size:
            h_step = max(1, h // max_size)
            w_step = max(1, w // max_size)
            class_map_display = class_map[::h_step, ::w_step]
        else:
            class_map_display = class_map
        
        # Création de la carte de classification
        fig = plt.figure(figsize=(10, 6))
        plt.imshow(
            class_map_display,
            cmap=plt.cm.colors.ListedColormap(self.color_maps['vegetation'][:len(classification_results['vegetation_types'])])
        )
        
        # Ajout de la légende
        patches = [plt.Rectangle((0, 0), 1, 1, fc=color)
                  for color in self.color_maps['vegetation'][:len(classification_results['vegetation_types'])]]
        plt.legend(
            patches,
            classification_results['vegetation_types'].values(),
            loc='center left',
            bbox_to_anchor=(1, 0.5)
        )
        
        plt.title("Classification de la Végétation")
        plt.colorbar(label="Classes de végétation")
        st.pyplot(fig)
        
        # Affichage des statistiques
        self._display_classification_stats(classification_results)
    
    def display_carbon_results(self, carbon_results):
        """Affiche les résultats de l'estimation du carbone."""
        # Sous-échantillonnage pour l'affichage si nécessaire
        max_size = 400  # Taille maximale pour l'affichage
        carbon_map = carbon_results['carbon_map']
        
        h, w = carbon_map.shape
        if h > max_size or w > max_size:
            h_step = max(1, h // max_size)
            w_step = max(1, w // max_size)
            carbon_map_display = carbon_map[::h_step, ::w_step]
        else:
            carbon_map_display = carbon_map
            
        # Carte de distribution du carbone
        fig = plt.figure(figsize=(10, 6))
        plt.imshow(carbon_map_display, cmap='viridis')
        plt.colorbar(label="Stock de carbone (tC/ha)")
        plt.title("Distribution Spatiale du Stock de Carbone")
        st.pyplot(fig)
        
        # Simplifier les données pour le graphique à barres
        # Limiter le nombre de classes si nécessaire
        carbon_by_class = carbon_results['carbon_by_class']
        if len(carbon_by_class) > 10:
            # Trouver les classes les plus importantes
            sorted_classes = sorted(carbon_by_class.items(), key=lambda x: x[1], reverse=True)
            top_classes = dict(sorted_classes[:9])
            other_carbon = sum(dict(sorted_classes[9:]).values())
            if other_carbon > 0:
                top_classes["Autres classes"] = other_carbon
            carbon_by_class = top_classes
        
        # Graphique de répartition par classe
        fig = px.bar(
            x=list(carbon_by_class.keys()),
            y=list(carbon_by_class.values()),
            labels={'x': 'Type de végétation', 'y': 'Stock de carbone (tC)'},
            title="Répartition du Stock de Carbone par Type de Végétation"
        )
        
        # Définir une taille fixe pour le graphique
        fig.update_layout(
            width=800,
            height=500,
            margin=dict(l=50, r=50, t=100, b=100)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistiques globales
        st.subheader("Statistiques Globales")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Stock total de carbone", f"{carbon_results['total_carbon']:.1f} tC")
        with col2:
            st.metric("Densité moyenne", f"{carbon_results['mean_density']:.1f} tC/ha")
        with col3:
            st.metric("Surface totale", f"{carbon_results['total_area']:.1f} ha")
    
    def _display_point_cloud_3d(self, las_data):
        """Affiche le nuage de points en 3D."""
        # Échantillonnage plus agressif des points
        max_points = 20000  # Réduit de 100000 à 20000
        if len(las_data.points) > max_points:
            indices = np.random.choice(len(las_data.points), max_points, replace=False)
            # Conversion explicite en tableaux NumPy standard
            x = np.array(las_data.x[indices])
            y = np.array(las_data.y[indices])
            z = np.array(las_data.z[indices])
            intensity = np.array(las_data.intensity[indices])
        else:
            # Conversion explicite en tableaux NumPy standard
            x = np.array(las_data.x)
            y = np.array(las_data.y)
            z = np.array(las_data.z)
            intensity = np.array(las_data.intensity)
        
        # Réduire la taille du rendu
        marker_size = 1  # Réduit de 2 à 1
        
        # Création du nuage de points 3D
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=marker_size,
                color=intensity,
                colorscale='Viridis',
                opacity=0.8
            )
        )])
        
        fig.update_layout(
            scene=dict(
                aspectmode='data'
            ),
            title="Nuage de Points LiDAR",
            # Réduire la taille du rendu
            width=800,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _display_height_distribution(self, las_data):
        """Affiche la distribution des hauteurs."""
        # Conversion explicite en tableau NumPy standard
        heights = np.array(las_data.z)
        
        fig = px.histogram(
            x=heights,
            nbins=50,
            labels={'x': 'Hauteur (m)', 'y': 'Nombre de points'},
            title="Distribution des Hauteurs"
        )
        st.plotly_chart(fig)
    
    def _display_point_density(self, las_data):
        """Affiche la densité des points."""
        # Conversion explicite en tableaux NumPy standard
        x_values = np.array(las_data.x)
        y_values = np.array(las_data.y)
        
        # Création d'une grille 2D pour la densité
        x_min, x_max = x_values.min(), x_values.max()
        y_min, y_max = y_values.min(), y_values.max()
        
        grid_size = 100
        x_bins = np.linspace(x_min, x_max, grid_size)
        y_bins = np.linspace(y_min, y_max, grid_size)
        
        density, _, _ = np.histogram2d(x_values, y_values, bins=[x_bins, y_bins])
        
        fig = px.imshow(
            density.T,
            labels=dict(x="X", y="Y", color="Densité de points"),
            title="Carte de Densité des Points"
        )
        st.plotly_chart(fig)
    
    def _create_surface_plot(self, data, title):
        """Crée un graphique de surface pour les modèles numériques."""
        # Réduire la résolution des données pour les visualisations
        # Sous-échantillonnage du modèle pour réduire la taille
        max_size = 100  # Taille maximale pour chaque dimension
        
        h, w = data.shape
        if h > max_size or w > max_size:
            # Calculer les pas d'échantillonnage
            h_step = max(1, h // max_size)
            w_step = max(1, w // max_size)
            data_reduced = data[::h_step, ::w_step]
        else:
            data_reduced = data
            
        # Assurez-vous que les données sont un tableau NumPy standard
        data_np = np.array(data_reduced)
        
        # Créer le graphique de surface avec une taille réduite
        fig = go.Figure(data=[go.Surface(z=data_np)])
        fig.update_layout(
            title=title,
            scene=dict(
                aspectmode='data'
            ),
            width=600,
            height=400
        )
        return fig
    
    def _display_classification_stats(self, classification_results):
        """Affiche les statistiques de classification."""
        st.subheader("Statistiques de Classification")
        
        # Calcul des statistiques par classe
        classification_map_np = np.array(classification_results['classification_map'])
        unique, counts = np.unique(classification_map_np, return_counts=True)
        total_pixels = counts.sum()
        
        # Création du tableau de statistiques
        stats_data = []
        for class_num, count in zip(unique, counts):
            if class_num in classification_results['vegetation_types']:
                stats_data.append({
                    "Classe": classification_results['vegetation_types'][class_num],
                    "Nombre de pixels": int(count),  # Conversion explicite en int standard
                    "Pourcentage": f"{(count/total_pixels)*100:.1f}%"
                })
        
        # Affichage du tableau
        st.table(stats_data)
