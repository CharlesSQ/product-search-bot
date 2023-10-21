import openai
import pinecone

openai.api_key = "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"
MODEL = "text-embedding-ada-002"
index_name = 'semantic-search-openai-labels'

pinecone.init(
    api_key="296bcfaf-160f-4244-b7bb-0a161e7d7459",
    environment="us-west1-gcp"
)

if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1536)

index = pinecone.Index(index_name)
