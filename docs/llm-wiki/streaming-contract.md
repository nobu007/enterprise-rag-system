# Streaming pipeline contract

`StreamingRAGPipeline.stream_query` and `StreamingRAGService.stream_query_response`
ignore provider stream chunks with no choices instead of indexing a missing choice.
After a clean stream end, each path emits its completion marker (`done` or
`is_done`); the service path also carries its final sources and timing metadata.

The live HTTP endpoint is `POST /api/v1/query/stream`. Clients send a JSON
`StreamingQueryRequest` body and receive `text/event-stream`; GET query-string
and `EventSource` examples are not valid for this route.

Evidence:

- `app/api/routes/query.py`
- `README.md`
- `app/services/rag_pipeline.py`
- `tests/unit/test_streaming_rag_pipeline.py`
- `tests/unit/test_query_stream_coverage.py`
- `tests/unit/test_api_docs.py`

Verification:

- `pytest tests/unit/test_streaming_rag_pipeline.py tests/unit/test_rag_pipeline.py -q`
  covers the streaming completion behavior.
- `pytest tests/unit/test_api_docs.py::TestAPIDocumentation::test_streaming_documentation_matches_post_json_contract -q`
  checks the mounted path, POST method, JSON request body, SSE response, and
  client examples.
