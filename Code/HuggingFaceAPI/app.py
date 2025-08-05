import gradio as gr # type: ignore
import pandas as pd
import re
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

weights = {
    "information_density": 0.1468,  # Increased from 0.1468
    "unique_key_points": 0.8323,    # Maintained high weight
    "strength_word_count": 0.0034,  # Reduced from 0.0034
    "weakness_word_count": 0.0047,  # Reduced from 0.0047
    "discussion_word_count": 0.0127  # Reduced from 0.0127
}

THRESHOLDS = {
  "normalized_length": (0.1510989010989011, 0.3722527472527472),
  "unique_key_points": (2.0, 5.0),
  "information_density": (0.00727669710202455, 0.011291644452510826),
  "unique_insights_per_word": 13.607476635514018,
  "composite_score": (4.5941101516587315, 10.581525294084457),
  "adjusted_argument_strength": 0.0616150390018116,
}

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

def segment_comments(comments):
    if comments == "N/A":
        return {"strengths": "", "weaknesses": "", "general_discussion": ""}
    
    strengths = re.search(r"- Strengths:\n([\s\S]*?)(\n- Weaknesses:|\Z)", comments)
    weaknesses = re.search(r"- Weaknesses:\n([\s\S]*?)(\n- General Discussion:|\Z)", comments)
    general_discussion = re.search(r"- General Discussion:\n([\s\S]*?)\Z", comments)
    
    return {
        "strengths": strengths.group(1).strip() if strengths else "",
        "weaknesses": weaknesses.group(1).strip() if weaknesses else "",
        "general_discussion": general_discussion.group(1).strip() if general_discussion else ""
    }

def preprocess(comment, abstract):
    df = pd.DataFrame({"comments": [comment]})
    abstracts = pd.DataFrame({"abstract": [abstract]})

    segmented_reviews = df["comments"].apply(segment_comments)
    df["strengths"] = segmented_reviews.apply(lambda x: x["strengths"])
    df["weaknesses"] = segmented_reviews.apply(lambda x: x["weaknesses"])
    df["general_discussion"] = segmented_reviews.apply(lambda x: x["general_discussion"])

    comments_embeddings = model.encode(df['comments'].tolist(), convert_to_tensor=True)
    abstract_embeddings = model.encode(abstracts["abstract"].tolist(), convert_to_tensor=True) 
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

    return df 

def calculate_composite_score(df):
    df['composite_score'] = (
        weights['information_density'] * df['information_density'] +
        weights['unique_key_points'] * df['unique_key_points'] +
        weights['strength_word_count'] * df['strength_word_count'] +
        weights['weakness_word_count'] * df['weakness_word_count'] +
        weights['discussion_word_count'] * df['discussion_word_count']
    )

    return df

def classify_review_quality(row):
    if row['composite_score'] > 10.581525294084457: 
        return 'Excellent'
    elif row['composite_score'] < 4.5941101516587315:  
        return 'Poor'
    else:
        return 'Moderate'  

def determine_review_quality(df):

    df['normalized_length'] = df['total_word_count'] / df['total_word_count'].max()
    df['unique_insights_per_word'] = df['unique_key_points'] / df['normalized_length']
    df['adjusted_argument_strength'] = df['argument_strength'] / (1 + df['sentence_complexity'])

    df['review_quality'] = df.apply(classify_review_quality, axis=1)

    return df

def heuristic_optimization(row):
    suggestions = []

    if row["strength_word_count"] > 100 and row["strength_argument_score"] < THRESHOLDS["adjusted_argument_strength"]:
        suggestions.append("Summarize redundant strengths.")
    elif row["strength_word_count"] < 50 and row["strength_argument_score"] < THRESHOLDS["adjusted_argument_strength"]:
        suggestions.append("Add more impactful strengths.")

    if row["weakness_word_count"] > 100 and row["weakness_argument_score"] < THRESHOLDS["adjusted_argument_strength"]:
        suggestions.append("Remove repetitive criticisms.")
    elif row["weakness_word_count"] < 50 and row["weakness_argument_score"] < THRESHOLDS["adjusted_argument_strength"]:
        suggestions.append("Add specific, actionable weaknesses.")

    if row["discussion_word_count"] < 100 and row["information_density"] < THRESHOLDS["information_density"][0]:
        suggestions.append("Elaborate with new insights or examples.")
    elif row["discussion_word_count"] > 300 and row["information_density"] > THRESHOLDS["information_density"][1]:
        suggestions.append("Summarize key discussion points.")

    if row["normalized_length"] < THRESHOLDS["normalized_length"][0]:
        suggestions.append("Expand sections for better coverage.")
    elif row["normalized_length"] > THRESHOLDS["normalized_length"][1]:
        suggestions.append("Condense content to improve readability.")

    if row["unique_key_points"] < THRESHOLDS["unique_key_points"][0]:
        suggestions.append("Add more unique insights.")
    elif row["unique_key_points"] > THRESHOLDS["unique_key_points"][1]:
        suggestions.append("Streamline ideas for clarity.")

    if row["review_quality"] == "Low":
        suggestions.append("Significant revisions required.")
    elif row["review_quality"] == "Moderate":
        suggestions.append("Minor refinements recommended.")

    return suggestions

def pipeline(comment, abstract):
    df = preprocess(comment, abstract)
    df = calculate_composite_score(df)
    df = determine_review_quality(df)
    df["optimization_suggestions"] = df.apply(heuristic_optimization, axis=1)
    return df["composite_score"][0], " ".join(df["optimization_suggestions"][0])

demo = gr.Interface(fn=pipeline, inputs=["text", "text"], outputs=["text", "text"])
demo.launch()
