---
Document ID: MD-TEST-2026
Author: Yasir Atiq, Principal ML Engineer
Last Updated: May 7, 2026
Classification: Public
---

# DATA INGESTION STANDARDS FOR KNOWLEDGE RETRIEVAL

## 1. Executive Summary

This document defines parsing parameters for markdown-formatted operational logs. The goal of this test file is to validate Markdown splitters (like `MarkdownTextSplitter` or `MarkdownHeaderTextSplitter`) to ensure they do not lose context, break code syntax, or drop embedded metadata during the RAG ingestion pipeline.

---

## 2. Infrastructure Parameters

### 2.1 Storage Configuration

The system relies on decentralized staging buckets. Ensure that chunks maintain spatial reference to these exact configurations:

* **Primary Bucket:** `s3://prod-rag-ingest-source-99/`
* **Sync Interval:** 15 minutes
* **Encryption standard:** AES-256-GCM
* **Maximum line length limit:** 10,000 characters

### 2.2 Supported Markdown Dialects

The chunker must successfully parse standard specifications without throwing delimiter errors:

1. **CommonMark:** Strict compliance required for text bodies.
2. **GitHub Flavored Markdown (GFM):** Required for parsing tables and autolinks.
3. **MDX:** Component tags should be ignored or preserved as raw strings.

---

## 3. Parsing Validation Matrix

Use this embedded table to check if your chunker preserves column alignments and data integrity across cell boundaries.

| Pipeline ID  | Target Tokenizer | Max Chunk Size | Overlap Ratio | Status     |
| :----------: | :--------------: | :------------: | :-----------: | :--------: |
| PL-DENSE-01  | cl100k_base      | 512 tokens     | 0.10          | Active     |
| PL-SPARSE-02 | p50k_base        | 1024 tokens    | 0.15          | Deprecated |
| PL-HYBRID-03 | o200k_base       | 256 tokens     | 0.20          | Testing    |

---

## 4. Code Block Extraction Tests

Code blocks must be treated as unified semantic blocks. The text splitter should not split a code block down the middle.

```python
# Validation hook for chunk integrity checks
def verify_markdown_payload(payload: dict) -> bool:
    required_keys = ["document_id", "author", "chunks"]
    if not all(key in payload for key in required_keys):
        raise ValueError("Missing critical metadata keys.")
    return len(payload["chunks"]) > 0
```

> **Warning on Tokenization:** Splitting inside the `verify_markdown_payload` function body will break semantic search continuity for engineering queries. Ensure syntax-aware splitting rules are active.

---

## 5. Retrieval Verification Questions

To verify that your Markdown ingestion pipeline works end-to-end, execute these targeted queries against your vector index after chunking and embedding are complete:

1. *What is the exact S3 bucket path used for primary storage configuration?*
   **Expected Answer:** `s3://prod-rag-ingest-source-99/` (Tests bullet point extraction)
2. *Which pipeline ID uses the o200k_base tokenizer and what is its status?*
   **Expected Answer:** `PL-HYBRID-03` and its status is `Testing` (Tests table cell value extraction)
3. *What are the three required keys validated in the python code block?*
   **Expected Answer:** `document_id`, `author`, and `chunks` (Tests multi-line code block parsing)
4. *Who is listed as the author of the markdown ingestion standards document?*
   **Expected Answer:** Yasir Atiq, Principal ML Engineer (Tests document header metadata parsing)
