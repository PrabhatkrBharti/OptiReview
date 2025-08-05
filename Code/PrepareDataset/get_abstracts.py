import json
import os
import pandas as pd

json_dir = "../../PeerRead/data/acl_2017/train/reviews/"

abstracts = []

for filename in os.listdir(json_dir):
    if filename.endswith(".json"):
        file_path = os.path.join(json_dir, filename)
        with open(file_path, "r") as file:
            data = json.load(file)
            reviews = data.get("reviews")
            review_count = len(reviews) if reviews else 0
            for i in range(review_count):
                abstracts.append({"abstract": data.get("abstract")})

df = pd.DataFrame(abstracts)

output_path = "../../Datasets/abstracts_list.csv"
df.to_csv(output_path, index=False)
print(f"Abstracts extracted and saved to {output_path}")
