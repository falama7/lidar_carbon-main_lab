import os
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import matplotlib.pyplot as plt
import numpy as np

class ReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        )
        self.subtitle_style = ParagraphStyle(
            'CustomSubTitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=20
        )
        self.normal_style = self.styles['Normal']
    
    def generate(self, carbon_results, include_maps=True, include_stats=True,
                include_methods=True, output_dir=None):
        """Génère un rapport PDF complet."""
        if output_dir is None:
            output_dir = os.getcwd()
            
        # Création du dossier de sortie s'il n'existe pas
        os.makedirs(output_dir, exist_ok=True)
        
        # Nom du fichier de rapport
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(output_dir, f"rapport_carbone_{timestamp}.pdf")
        
        # Création du document
        doc = SimpleDocTemplate(
            report_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Contenu du rapport
        story = []
        
        # Titre
        story.append(Paragraph("Rapport d'Estimation du Stock de Carbone", self.title_style))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}", self.normal_style))
        story.append(Spacer(1, 30))
        
        # Résumé des résultats
        story.append(Paragraph("Résumé des Résultats", self.subtitle_style))
        summary_data = [
            ["Métrique", "Valeur"],
            ["Stock total de carbone", f"{carbon_results['total_carbon']:.1f} tC"],
            ["Densité moyenne", f"{carbon_results['mean_density']:.1f} tC/ha"],
            ["Surface totale", f"{carbon_results['total_area']:.1f} ha"]
        ]
        summary_table = Table(summary_data, colWidths=[4*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        if include_stats:
            # Détail par classe de végétation
            story.append(Paragraph("Répartition par Type de Végétation", self.subtitle_style))
            class_data = [["Type de Végétation", "Stock de Carbone (tC)", "Proportion (%)"]]
            total_carbon = carbon_results['total_carbon']
            
            for veg_type, carbon in carbon_results['carbon_by_class'].items():
                proportion = (carbon / total_carbon * 100) if total_carbon > 0 else 0
                class_data.append([
                    veg_type,
                    f"{carbon:.1f}",
                    f"{proportion:.1f}%"
                ])
            
            class_table = Table(class_data, colWidths=[3*inch, 2*inch, 2*inch])
            class_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(class_table)
            story.append(Spacer(1, 20))
        
        if include_maps:
            # Génération et sauvegarde des cartes
            story.append(Paragraph("Cartographie du Stock de Carbone", self.subtitle_style))
            
            # Carte de distribution du carbone
            plt.figure(figsize=(8, 6))
            plt.imshow(carbon_results['carbon_map'], cmap='viridis')
            plt.colorbar(label="Stock de carbone (tC/ha)")
            plt.title("Distribution Spatiale du Stock de Carbone")
            map_path = os.path.join(output_dir, "carte_carbone.png")
            plt.savefig(map_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Ajout de la carte au rapport
            story.append(Image(map_path, width=6*inch, height=4*inch))
            story.append(Spacer(1, 20))
        
        if include_methods:
            # Description des méthodes
            story.append(Paragraph("Méthodologie", self.subtitle_style))
            story.append(Paragraph("""
            L'estimation du stock de carbone a été réalisée en utilisant une approche combinant :
            
            1. Analyse LiDAR : Les données LiDAR ont été utilisées pour créer un modèle de hauteur de canopée (CHM) précis.
            
            2. Classification de la végétation : La végétation a été classifiée en différentes catégories basées sur la hauteur.
            
            3. Estimation du carbone : Le stock de carbone a été estimé en utilisant des équations allométriques et/ou des modèles de machine learning, prenant en compte les caractéristiques spécifiques de chaque type de végétation.
            
            Les coefficients utilisés pour l'estimation sont basés sur des études scientifiques et peuvent être ajustés selon les spécificités locales.
            """, self.normal_style))
        
        # Génération du PDF
        doc.build(story)
        
        return report_path 