import chromadb
chroma_client = chromadb.HttpClient(host='127.0.0.1', port=4810)
print(chroma_client.list_collections())