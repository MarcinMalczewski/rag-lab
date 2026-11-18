# Test questions for `sample/documents.jsonl`

Questions for comparing chunking profiles and retrieval approaches on
[documents.jsonl](documents.jsonl). Each row gives the expected document(s), the
fact that answers the question, and what a difference in results would reveal.

Pass criterion: an expected document appears in the top-k (k = 5 suggested) and
the retrieved chunk contains the answer. Record the rank of the first correct
chunk for each question and setup.

## How to read the results

| Approach | Expected to win on |
|---|---|
| Small window (`naive-100-0`) | Precise, single-fact lookups (groups A, C). Loses answers split across chunks. |
| Large window with overlap (`naive-500-100`) | Questions needing several sentences (group A). Documents are only ~400–500 tokens, so this is close to one chunk per document. |
| Late chunking | Chunks that are ambiguous without the rest of the document (group B). |
| Dense retrieval | Paraphrased questions with little word overlap (group D). |
| Lexical / hybrid (BM25) | Exact identifiers, codes and rare terms (group C). |
| Reranking | Multi-document and near-miss questions (groups E, F), where the right chunk is retrieved but ranked low. |

## A. Answers spanning several sentences (chunk size and overlap)

| ID | Question | Expected | Answer / what it tests |
|---|---|---|---|
| sample-A1 | What are the two stages of photosynthesis and what does each one produce? | doc-010 | Light-dependent reactions give oxygen, ATP and NADPH; the Calvin cycle builds sugars. The answer covers a ~110-word paragraph, so 100-token windows may return only one stage. |
| sample-A2 | Who first reached the summit of Everest, when, and as part of which expedition? | doc-007 | Hillary and Tenzing Norgay, 29 May 1953, the British expedition led by John Hunt. Date, names and leader are in separate sentences. |
| sample-A3 | What are the trade-offs between small and large chunks? | doc-002 | Small chunks are precise but lose context; large chunks keep context but dilute the signal and use more of the LLM context. Spread over two paragraphs. |
| sample-A4 | What steps should a client take after getting a concurrent update conflict? | doc-009 | ERR_4091: fetch the order again, apply the change, send the update with the new version in `If-Match`. Tests a multi-step answer. |

## B. Chunks that lose meaning without the surrounding document (late chunking)

The expected paragraph does not name its subject, so a chunk embedded alone is
ambiguous.

| ID | Question | Expected | Answer / what it tests |
|---|---|---|---|
| sample-B1 | What problems has the growing number of climbers caused on Everest? | doc-007 | Queues near the summit, waste, debate about Sherpa support. The final paragraph never contains the word "Everest". |
| sample-B2 | What limits the rate of photosynthesis? | doc-010 | Light intensity, carbon dioxide concentration and temperature. The final paragraph never contains the word "photosynthesis". |
| sample-B3 | Why might a small recall difference between two RAG setups not come from chunking? | doc-001 | Approximate search (HNSW) adds its own noise; fix index settings or compare with brute force. The paragraph never mentions "HNSW" by name. |
| sample-B4 | When are neighbouring hits merged before the LLM sees them? | doc-006 | A merge policy combines adjacent chunks of the same document into one longer passage. |

## C. Exact identifiers and rare terms (lexical / hybrid should win)

| ID | Question | Expected | Answer / what it tests |
|---|---|---|---|
| sample-C1 | What does ERR_4012 mean? | doc-009 | Expired token; refresh via `/auth/refresh`. Dense models often confuse similar codes (ERR_4001, ERR_4034). |
| sample-C2 | Which error code is safe to retry with the same Idempotency-Key header? | doc-009 | ERR_5003. Rare header name. |
| sample-C3 | Which header should be sent to support to find a request in the logs? | doc-009 | `X-Request-Id`. |
| sample-C4 | What do `ef_construct` and `m` control in HNSW? | doc-001 | `m`: links per node; `ef_construct`: candidates examined while building. |
| sample-C5 | What constant is traditionally used in Reciprocal Rank Fusion? | doc-004 | 60. |

## D. Paraphrased questions with little word overlap (dense should win)

| ID | Question | Expected | Answer / what it tests |
|---|---|---|---|
| sample-D1 | How do green organisms make sugar from sunshine? | doc-010 | Photosynthesis. No shared keywords with the text. |
| sample-D2 | My client keeps being refused because it sends too many calls each minute. | doc-009 | ERR_4290: 100 requests per minute, honour `Retry-After`, use exponential backoff. |
| sample-D3 | Why does freshly roasted coffee bubble when hot water first touches it? | doc-008 | The bloom releases carbon dioxide; wait 30–45 seconds. |
| sample-D4 | Why does my brew taste sour and watery? | doc-008 | Under-extraction: grind too coarse. |
| sample-D5 | How can a model understand a sentence that refers to something mentioned earlier in the text? | doc-003 | Late chunking gives each chunk the context of the whole document. |

## E. Near-miss and confusable questions (reranking)

| ID | Question | Expected | Answer / what it tests |
|---|---|---|---|
| sample-E1 | How many dimensions does nomic-embed-text produce? | doc-005 | 768. Several other documents mention dimensions, models or vectors. |
| sample-E2 | Which reranker model type reads the query and passage together? | doc-006 | A cross-encoder. Competes with doc-005 and doc-004 on retrieval vocabulary. |
| sample-E3 | Who climbed Everest without supplementary oxygen? | doc-007 | Messner and Habeler, 1978. The first-ascent paragraph is a stronger lexical match but is the wrong answer. |

## F. Answers needing several documents

| ID | Question | Expected | Answer / what it tests |
|---|---|---|---|
| sample-F1 | Which techniques improve retrieval beyond a single dense vector search? | doc-004, doc-006 (also doc-003) | Hybrid BM25 plus dense with fusion, and cross-encoder reranking. Measures whether all relevant documents reach the top-k. |
| sample-F2 | Which settings must be recorded or fixed so that experiments are comparable? | doc-001, doc-003, doc-005, doc-006 | HNSW parameters, embedding model and revision, pooling, reranker settings. |

## G. Out-of-corpus (should not be answered)

| ID | Question | Expected | Answer / what it tests |
|---|---|---|---|
| sample-G1 | What is the capital of France? | none | Top scores should be clearly lower than for in-corpus questions. Useful for choosing a score threshold. |
| sample-G2 | How do I reset my Orders API password? | none | Overlaps with doc-009 vocabulary but has no answer. Tests false positives. |

## Suggested reading of a comparison

- A1, A2, A4 improving from `naive-100-0` to `naive-500-100` shows the cost of
  splitting answers across chunks.
- B1–B4 improving under late chunking, but not A or C, is the specific benefit
  of contextual embeddings.
- C-questions failing under dense-only retrieval and passing under hybrid shows
  the value of the lexical component; D-questions should show the opposite.
- E-questions improving after reranking, with a measured latency cost, is the
  reranking trade-off.
