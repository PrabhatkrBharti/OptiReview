import json
import os
import pandas as pd

json_dir = "../../PeerRead/data/acl_2017/train/reviews/"

processed_reviews = []

for filename in os.listdir(json_dir):
    if filename.endswith(".json"):
        file_path = os.path.join(json_dir, filename)
        with open(file_path, "r") as file:
            data = json.load(file)
            for review in data["reviews"]:
                processed_reviews.append({
                    "IMPACT": review.get("IMPACT"),
                    "SUBSTANCE": review.get("SUBSTANCE"),
                    "APPROPRIATENESS": review.get("APPROPRIATENESS"),
                    "SOUNDNESS_CORRECTNESS": review.get("SOUNDNESS_CORRECTNESS"),
                    "ORIGINALITY": review.get("ORIGINALITY"),
                    "RECOMMENDATION": review.get("RECOMMENDATION"),
                    "CLARITY": review.get("CLARITY"),
                    "REVIEWER_CONFIDENCE": review.get("REVIEWER_CONFIDENCE"),
                    "PRESENTATION_FORMAT": review.get("PRESENTATION_FORMAT"),
                    "is_meta_review": review.get("is_meta_review"),
                    "comments": review.get("comments")
                })

df = pd.DataFrame(processed_reviews)

output_path = "../../Datasets/processed_reviews.csv"
df.to_csv(output_path, index=False)
print(f"Extracted data saved to {output_path}")
