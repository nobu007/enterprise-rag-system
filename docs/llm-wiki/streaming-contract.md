# Streaming pipeline contract

`StreamingRAGPipeline.stream_query` and `StreamingRAGService.stream_query_response`
ignore provider stream chunks with no choices instead of indexing a missing choice.
After a clean stream end, each path emits its completion marker (`done` or
`is_done`); the service path also carries its final sources and timing metadata.

Evidence:

- `app/services/rag_pipeline.py`
- `tests/unit/test_streaming_rag_pipeline.py`
- `tests/unit/test_query_stream_coverage.py`

Verification: `pytest tests/unit/test_streaming_rag_pipeline.py tests/unit/test_rag_pipeline.py -q`
passed after the guard and regression test were added.
