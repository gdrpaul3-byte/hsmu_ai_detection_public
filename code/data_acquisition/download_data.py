import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd

SERVICE_ACCOUNT_PATH = "YOUR_FIREBASE_ADMIN_SDK_JSON_PATH"

# --- 1. Firebase Admin SDK setup ---
# Set SERVICE_ACCOUNT_PATH to your local Firebase Admin SDK JSON file.
try:
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred)
    print("Firebase initialized successfully.")
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    print("Replace SERVICE_ACCOUNT_PATH with your Firebase Admin SDK JSON path.")
    exit()

# Firestore database client
db = firestore.client()


def download_collection_to_csv(collection_name):
    """
    Download all documents in a Firestore collection and save them as a CSV file.
    """
    print(f"\nDownloading collection: '{collection_name}'...")

    try:
        docs_ref = db.collection(collection_name).stream()
        docs_list = [doc.to_dict() for doc in docs_ref]

        if not docs_list:
            print(f"Warning: No documents found in '{collection_name}'.")
            return

        df = pd.DataFrame(docs_list)
        file_name = f"{collection_name}_export.csv"
        df.to_csv(file_name, index=False, encoding="utf-8-sig")

        print(f"Success: downloaded {len(df)} documents to '{file_name}'")

    except Exception as e:
        print(f"An error occurred while downloading '{collection_name}': {e}")


if __name__ == "__main__":
    download_collection_to_csv("surveys")
    download_collection_to_csv("responses")
