import pickle
import re
from flask import Flask, request, jsonify, render_template, send_file
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from googleapiclient.discovery import build
import pandas as pd
import matplotlib.pyplot as plt
import os

# Initialisation Flask
app = Flask(__name__)

# Charger les modèles sauvegardés
with open('text_vectorizer.sav', 'rb') as f:
    loaded_vectorizer = pickle.load(f)
with open('trained_model.sav', 'rb') as f:
    loaded_model = pickle.load(f)


api_key = "AIzaSyDuuKa4rOXUUAWp1V-wA2JpAV4mIr6obUs"
youtube = build("youtube", "v3", developerKey=api_key)


port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

def obtenir_meta_video(video_id):
    """
    Obtenir le titre et la miniature de la vidéo.
    """
    try:
        response = youtube.videos().list(
            part="snippet",
            id=video_id
        ).execute()

        if response['items']:
            snippet = response['items'][0]['snippet']
            titre = snippet['title']
            thumbnail_url = snippet['thumbnails']['high']['url']  # Miniature de haute qualité
            return titre, thumbnail_url
        else:
            return None, None
    except Exception as e:
        raise ValueError(f"Impossible de récupérer les métadonnées de la vidéo : {str(e)}")


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    youtube_url = request.form.get('youtube_url')
    if not youtube_url:
        return jsonify({"error": "L'URL de la vidéo est manquante"}), 400

    try:
        # Extraire l'ID de la vidéo
        video_id = youtube_url.split('v=')[-1][:11]

        # Récupérer les métadonnées de la vidéo
        titre, thumbnail_url = obtenir_meta_video(video_id)
        if not titre or not thumbnail_url:
            return jsonify({"error": "Impossible de récupérer les métadonnées de la vidéo."}), 400

        # Récupérer les commentaires
        commentaires = obtenir_commentaires(video_id)
        commentaires_textes = [comment["Commentaire"] for comment in commentaires]

        # Prétraitement des commentaires
        processed_comments = [stemming(comment) for comment in commentaires_textes]
        comments_vectors = loaded_vectorizer.transform(processed_comments)

        # Prédictions
        predictions = loaded_model.predict(comments_vectors)
        positive_count = int(sum(predictions == 1))  # Convertir numpy.int64 en int
        negative_count = int(sum(predictions == 0))

        # Générer un graphique
        labels = ['Positifs', 'Négatifs']
        sizes = [positive_count, negative_count]
        colors = ['#4CAF50', '#F44336']
        explode = (0.1, 0)

        plt.figure(figsize=(6, 6))
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title("Analyse des Sentiments")
        plt.axis('equal')

        # Enregistrer le graphique
        chart_path = os.path.join('static', 'sentiment_pie_chart.png')
        plt.savefig(chart_path)
        plt.close()

        # Retourner les résultats
        return jsonify({
            "title": titre,
            "thumbnail_url": thumbnail_url,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "chart_url": '/' + chart_path
        })

    except ValueError as ve:
        return jsonify({"error": f"Erreur dans les données : {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"error": f"Une erreur est survenue : {str(e)}"}), 500

def obtenir_commentaires(video_id, max_results=100):
    commentaires = []
    next_page_token = None

    while True:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            pageToken=next_page_token
        ).execute()

        for item in response['items']:
            commentaire = item['snippet']['topLevelComment']['snippet']['textDisplay']
            auteur = item['snippet']['topLevelComment']['snippet']['authorDisplayName']
            commentaires.append({"Auteur": auteur, "Commentaire": commentaire})

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break

    return commentaires

if __name__ == '__main__':
    app.run(debug=True)