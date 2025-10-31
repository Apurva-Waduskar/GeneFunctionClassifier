import requests
import pandas as pd
from tqdm import tqdm
import time

# =============================
# Fetch GO annotated proteins from UniProt
# =============================

def fetch_uniprot_go_data(size=500):
    """
    Fetch UniProt entries with GO annotations (Swiss-Prot reviewed only).
    """
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    query = "reviewed:true AND go:*"
    params = {
        "query": query,
        "fields": "accession,sequence,go_id",
        "format": "tsv",
        "size": 100
    }

    rows = []
    fetched = 0

    print(f"Fetching {size} proteins from UniProt with GO annotations...")
    while fetched < size:
        params["offset"] = fetched
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            print(f"Failed at offset {fetched}, retrying...")
            time.sleep(5)
            continue

        lines = response.text.strip().split("\n")[1:]
        for line in lines:
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            acc, seq, go_terms = parts
            rows.append((acc, seq, go_terms))
        fetched += len(lines)
        time.sleep(0.3)  # polite delay for UniProt API

    df = pd.DataFrame(rows, columns=["Protein_ID", "Sequence", "GO_Terms"])
    df = df.dropna(subset=["Sequence", "GO_Terms"])
    df.to_csv("data/go_subset.csv", index=False)
    print(f"\n✅ Saved real dataset: data/go_subset.csv ({len(df)} rows)")
    return df


if __name__ == "__main__":
    fetch_uniprot_go_data(500)
