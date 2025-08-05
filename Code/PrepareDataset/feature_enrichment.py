import pandas as pd
import spacy # type: ignore
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util # type: ignore
from transformers import pipeline, AutoTokenizer
import textstat # type: ignore

sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

nlp = spacy.load("en_core_web_sm")

model = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_csv("../../Datasets/segmented_reviews.csv")
abstracts_list = pd.read_csv("../../Datasets/abstracts_list.csv")
df = df.dropna(subset=['comments'])
abstracts_list = abstracts_list.fillna("No abstract provided.")

def chunk_text(text, max_length):
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"].squeeze(0).tolist()
    return [tokenizer.decode(tokens[i:i+max_length]) for i in range(0, len(tokens), max_length)]

def analyze_text(texts):
    results = []
    for text in texts:
        chunks = chunk_text(text, max_length=200)
        chunk_results = sentiment_analyzer(chunks)  
        overall_sentiment = {
            "label": "POSITIVE" if sum(1 for res in chunk_results if res["label"] == "POSITIVE") >= len(chunk_results) / 2 else "NEGATIVE",
            "score": sum(res["score"] for res in chunk_results) / len(chunk_results),
        }
        results.append(overall_sentiment)
    return results

def word_count(text):
    return len(text.split()) if isinstance(text, str) else 0

def count_citations(text):
    doc = nlp(text)
    return sum(1 for ent in doc.ents if ent.label_ in ['WORK_OF_ART', 'ORG', 'GPE'])

def calculate_unique_insights_per_word(text):
    sentences = text.split('.')  
    tfidf = TfidfVectorizer().fit_transform(sentences)
    similarities = cosine_similarity(tfidf)
    avg_similarity = (similarities.sum() - len(sentences)) / (len(sentences)**2 - len(sentences)) 
    return 1 - avg_similarity 

def calculate_unique_key_points_and_density(texts):
    unique_key_points = []
    information_density = []

    for text in texts:
        if not isinstance(text, str) or text.strip() == "":
            unique_key_points.append(0)
            information_density.append(0)
            continue

        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]

        embeddings = model.encode(sentences)

        n_clusters = max(1, len(sentences) // 5)  
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(embeddings)

        cluster_centers = kmeans.cluster_centers_
        unique_points_count = len(cluster_centers)

        word_count = len(text.split())
        density = unique_points_count / word_count if word_count > 0 else 0

        unique_key_points.append(unique_points_count)
        information_density.append(density)

    return unique_key_points, information_density

comments_embeddings = model.encode(df['comments'].tolist(), convert_to_tensor=True)
abstract_embeddings = model.encode(abstracts_list["abstract"].tolist(), convert_to_tensor=True) 
df['content_relevance'] = util.cos_sim(comments_embeddings, abstract_embeddings).diagonal()

df['evidence_support'] = df['comments'].apply(count_citations)

df['strengths'] = df['strengths'].fillna('').astype(str)
texts = df['strengths'].tolist()  
results = analyze_text(texts)  
df['strength_argument_score'] = [result['score'] for result in results]

df['weaknesses'] = df['weaknesses'].fillna('').astype(str)
texts = df['weaknesses'].tolist()  
results = analyze_text(texts)  
df['weakness_argument_score'] = [result['score'] for result in results]

df['argument_strength'] = (df['strength_argument_score'] + df['weakness_argument_score']) / 2

df['readability_index'] = df['comments'].apply(textstat.flesch_reading_ease)
df['sentence_complexity'] = df['comments'].apply(textstat.sentence_count)
df['technical_depth'] = df['readability_index'] / df['sentence_complexity']

df['total_word_count'] = df['comments'].apply(word_count)
df['strength_word_count'] = df['strengths'].apply(word_count)
df['weakness_word_count'] = df['weaknesses'].apply(word_count)
df['discussion_word_count'] = df['general_discussion'].apply(word_count)

average_length = df['total_word_count'].mean()
df['normalized_length'] = df['total_word_count'] / average_length
df["unique_key_points"], df["information_density"] = calculate_unique_key_points_and_density(df["comments"])

df['unique_insights_per_word'] = df['comments'].apply(calculate_unique_insights_per_word) / df['total_word_count']

df.to_csv('../../Datasets/final_dataset.csv', index=False)
print("Multiple features added and final dataset saved to 'final_dataset.csv'.")
