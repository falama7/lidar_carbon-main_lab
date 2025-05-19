FROM python:3.10-slim

WORKDIR /app

# Installation des dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    libgdal-dev \
    python3-dev \
    wget \
    unzip \
    ca-certificates \
    # Ajout de packages pour le débogage
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Configuration de WhiteboxTools pour Python
ENV PYTHONPATH=/usr/local/lib/python3.10/site-packages

# Installation de WhiteboxTools avec vérification
RUN wget --no-check-certificate https://www.whiteboxgeo.com/WBT_Linux/WhiteboxTools_linux_amd64.zip \
    && unzip WhiteboxTools_linux_amd64.zip \
    && ls -la WhiteboxTools_linux_amd64/WBT/ \
    && mv WhiteboxTools_linux_amd64/WBT/whitebox_tools /usr/local/bin/ \
    && chmod +x /usr/local/bin/whitebox_tools \
    && rm -rf WhiteboxTools_linux_amd64.zip WhiteboxTools_linux_amd64

ENV WBT_DIR="/usr/local/bin"

# Vérification que l'exécutable WhiteboxTools existe
RUN ls -la /usr/local/bin/whitebox_tools && \
    /usr/local/bin/whitebox_tools --help

# Création des répertoires nécessaires avec les bonnes permissions
RUN mkdir -p /usr/local/lib/python3.10/site-packages/whitebox/WBT/img \
    && mkdir -p /app/temp \
    && chmod -R 777 /app/temp \
    && chmod -R 777 /usr/local/lib/python3.10/site-packages/whitebox

# Copie des fichiers de requirements et installation des dépendances
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir scipy

# Copie du code de l'application
COPY . .

# Définition des variables d'environnement pour les chemins
ENV TEMP_DIR="/app/temp"
ENV DATA_DIR="/app/data"

# Création des répertoires de données avec les bonnes permissions
RUN mkdir -p ${TEMP_DIR} ${DATA_DIR} \
    && chmod -R 777 ${TEMP_DIR} ${DATA_DIR}

# Test que WhiteboxTools fonctionne dans l'environnement Python
RUN python -c "import whitebox; wbt = whitebox.WhiteboxTools(); print(wbt.version())"

# Exposition du port Streamlit
EXPOSE 8501

# Commande pour démarrer l'application
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
