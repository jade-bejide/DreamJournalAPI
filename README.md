# DreamJournal-API
Introducing DreamJournal-API, A Python REST-API built using FastAPI and hosting using fly.io. The sole endpoint (/entry) uses a Latent Dirichlet Allocation [1] trained model to infer the underlying emotion of a dream journal excerpt.

## Features
- **Stateless**: This API does not store dream journal excerpts and additionally works over HTTPS keeping your dream journal excerpts confidential

## Technology Stack
- Language: Python
- Environment: fly.io
- API Framework: FastAPI
- Third Party Libraries: gensim, nltk

## Documentation
### Dream Journal Entries
Get dream journal entry interpretation

`POST https://<fly.io URL>/entry`

Request Body:
```json
{
  "contents": "<Your Dream Journal Entry>"
}
```

## References
- [1] Blei, D.M., Ng, A.Y. and Jordan, M.I., 2003. Latent dirichlet allocation. Journal of machine Learning research, 3(Jan), pp.993-1022.
