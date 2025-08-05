import re
import pandas as pd

df = pd.read_csv("../../Datasets/cleaned_reviews.csv")

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

segmented_reviews = df["comments"].apply(segment_comments)

df["strengths"] = segmented_reviews.apply(lambda x: x["strengths"])
df["weaknesses"] = segmented_reviews.apply(lambda x: x["weaknesses"])
df["general_discussion"] = segmented_reviews.apply(lambda x: x["general_discussion"])

segmented_output_path = "../../Datasets/segmented_reviews.csv"
df.to_csv(segmented_output_path, index=False)
print(f"Segmented data saved to {segmented_output_path}")
