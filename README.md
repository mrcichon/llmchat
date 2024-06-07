currently houses a minimal implementation of RAG, along with the beginings of not-so-minimal one.
different base models are being tested on [TruthfulQA](https://github.com/sylinrl/TruthfulQA), with different embedding models being tested on [LEPISZCZE](https://arxiv.org/abs/2211.13112) and [KLEJ](https://arxiv.org/abs/2005.00630)

instructions to run
docker run -p 6333:6333 -v ~/path/to/loca/vector/database:/qdrant/storage:z qdrant/qdrant
ollama create newmodelname --file path/to/model/file
uvicorn app:app --reload
