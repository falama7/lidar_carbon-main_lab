import numpy as np
from sklearn.ensemble import RandomForestRegressor

class CarbonEstimator:
    def __init__(self):
        # Coefficients allométriques par défaut (à adapter selon les espèces)
        self.allometric_coeffs = {
            "Végétation basse": 0.5,    # tC/ha par mètre de hauteur
            "Arbustes": 2.0,            # tC/ha par mètre de hauteur
            "Arbres moyens": 5.0,       # tC/ha par mètre de hauteur
            "Grands arbres": 8.0        # tC/ha par mètre de hauteur
        }
        
        # Modèle ML (à entraîner si nécessaire)
        self.ml_model = None
    
    def estimate(self, classification_results, method="Allométrique"):
        """Estime le stock de carbone selon la méthode choisie."""
        if method == "Allométrique":
            return self._estimate_allometric(classification_results)
        elif method == "Machine Learning":
            return self._estimate_ml(classification_results)
        else:  # Hybride
            return self._estimate_hybrid(classification_results)
    
    def _estimate_allometric(self, classification_results):
        """Estimation du carbone par méthode allométrique."""
        # Récupération des données
        chm = classification_results['classification_map']
        veg_types = classification_results['vegetation_types']
        centers = classification_results['cluster_centers']
        
        # Initialisation de la carte de carbone
        carbon_map = np.zeros_like(chm, dtype=float)
        carbon_by_class = {}
        
        # Calcul pour chaque classe
        for class_num, veg_type in veg_types.items():
            if class_num == 0:  # Pas de végétation
                continue
                
            # Masque pour la classe actuelle
            mask = chm == class_num
            
            # Hauteur moyenne pour cette classe
            height = centers[class_num - 1]  # -1 car class_num commence à 1
            
            # Détermination du coefficient
            if height < 1:
                coeff = self.allometric_coeffs["Végétation basse"]
            elif height < 5:
                coeff = self.allometric_coeffs["Arbustes"]
            elif height < 15:
                coeff = self.allometric_coeffs["Arbres moyens"]
            else:
                coeff = self.allometric_coeffs["Grands arbres"]
            
            # Calcul du carbone
            carbon = height * coeff
            carbon_map[mask] = carbon
            
            # Stockage des résultats par classe
            carbon_by_class[veg_type] = float(np.sum(carbon_map[mask]))
        
        # Calcul des statistiques globales
        pixel_area = 1.0  # m² (à ajuster selon la résolution)
        total_area = np.sum(chm > 0) * pixel_area / 10000  # conversion en hectares
        total_carbon = float(np.sum(carbon_map))
        mean_density = total_carbon / total_area if total_area > 0 else 0
        
        return {
            'carbon_map': carbon_map,
            'carbon_by_class': carbon_by_class,
            'total_carbon': total_carbon,
            'mean_density': mean_density,
            'total_area': total_area
        }
    
    def _estimate_ml(self, classification_results):
        """Estimation du carbone par apprentissage automatique."""
        # Pour l'exemple, nous utilisons une approche simplifiée
        # En pratique, le modèle devrait être entraîné sur des données de terrain
        
        if self.ml_model is None:
            # Création et entraînement d'un modèle simple
            self.ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            # Simulation de données d'entraînement
            # En pratique, ces données devraient venir de mesures terrain
            X_train = classification_results['cluster_centers'].reshape(-1, 1)
            y_train = np.array([h * 2.5 for h in classification_results['cluster_centers']])
            self.ml_model.fit(X_train, y_train)
        
        # Prédiction pour chaque pixel
        chm = classification_results['classification_map']
        heights = classification_results['cluster_centers']
        carbon_map = np.zeros_like(chm, dtype=float)
        carbon_by_class = {}
        
        for class_num, veg_type in classification_results['vegetation_types'].items():
            if class_num == 0:
                continue
                
            mask = chm == class_num
            height = heights[class_num - 1]
            carbon = float(self.ml_model.predict([[height]])[0])
            carbon_map[mask] = carbon
            carbon_by_class[veg_type] = float(np.sum(carbon_map[mask]))
        
        # Calcul des statistiques globales
        pixel_area = 1.0
        total_area = np.sum(chm > 0) * pixel_area / 10000
        total_carbon = float(np.sum(carbon_map))
        mean_density = total_carbon / total_area if total_area > 0 else 0
        
        return {
            'carbon_map': carbon_map,
            'carbon_by_class': carbon_by_class,
            'total_carbon': total_carbon,
            'mean_density': mean_density,
            'total_area': total_area
        }
    
    def _estimate_hybrid(self, classification_results):
        """Estimation hybride combinant méthodes allométrique et ML."""
        # Obtention des deux estimations
        allometric_results = self._estimate_allometric(classification_results)
        ml_results = self._estimate_ml(classification_results)
        
        # Moyenne pondérée des estimations
        # On pourrait ajuster ces poids selon la confiance dans chaque méthode
        w_allometric = 0.6
        w_ml = 0.4
        
        carbon_map = (
            w_allometric * allometric_results['carbon_map'] +
            w_ml * ml_results['carbon_map']
        )
        
        carbon_by_class = {}
        for veg_type in classification_results['vegetation_types'].values():
            if veg_type in allometric_results['carbon_by_class']:
                carbon_by_class[veg_type] = (
                    w_allometric * allometric_results['carbon_by_class'][veg_type] +
                    w_ml * ml_results['carbon_by_class'][veg_type]
                )
        
        total_carbon = float(np.sum(carbon_map))
        total_area = allometric_results['total_area']  # même surface dans les deux cas
        mean_density = total_carbon / total_area if total_area > 0 else 0
        
        return {
            'carbon_map': carbon_map,
            'carbon_by_class': carbon_by_class,
            'total_carbon': total_carbon,
            'mean_density': mean_density,
            'total_area': total_area
        } 