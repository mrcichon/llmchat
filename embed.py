from qdrant_client import models, QdrantClient
from qdrant_client.http.models import PointStruct
from sentence_transformers import SentenceTransformer
import pdfplumber


encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
client = QdrantClient(path="embeddings")


def get_text(path):
    fulltext = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            fulltext += page.extract_text()
    print(fulltext)
    return fulltext


def chunking(text, size=500):
    chunks = []
    while len(text) > size:
        last_period_index = text[:size].rfind('.')
        if last_period_index == -1:
            last_period_index = size
        chunks.append(text[:last_period_index])
        text = text[last_period_index+1:]
    chunks.append(text)
    return chunks

def embed(chunks):
    points = []
    i = 1
    for chunk in chunks:
        i += 1
        print("Embeddings chunk:", chunk)
        points.append(PointStruct(id=i, vector=encoder.encode(chunk), payload={"text": chunk}))
    return points


def addtodb(points):
    client.upload_points(collection_name ="mytest1", points = points)



client.recreate_collection(
    collection_name="mytest1",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),
        distance=models.Distance.COSINE,
    ),
)



addtodb(embed(chunking(get_text("test.pdf"))))

addtodb(embed(chunking(get_text("test2.pdf"))))


hits = client.search(collection_name="mytest1", query_vector=encoder.encode("politechnika łódzka").tolist() , limit=1)
print(hits)



