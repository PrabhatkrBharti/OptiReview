import pandas as pd

df = pd.read_csv("../../Datasets/processed_reviews.csv")

df.drop(columns=["is_meta_review", "PRESENTATION_FORMAT"], inplace=True)

score_columns = ["IMPACT", "SUBSTANCE", "APPROPRIATENESS", "SOUNDNESS_CORRECTNESS",
                 "ORIGINALITY", "RECOMMENDATION", "CLARITY", "REVIEWER_CONFIDENCE"]

for col in score_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce") 
    df[col] = df[col] / df[col].max()

df.fillna("N/A", inplace=True) 

cleaned_output_path = "../../Datasets/cleaned_reviews.csv"
df.to_csv(cleaned_output_path, index=False)
print(f"Cleaned data saved to {cleaned_output_path}")
