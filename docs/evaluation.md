# Evaluation

`raglab evaluate` measures how well a setup retrieves the answers to a set of
labelled questions, so different setups (chunk sizes, embedding models, later
hybrid search or reranking) can be compared on the same questions. The command
itself is described in [CLI](cli.md#evaluation).

## How it works

- **Labels:** each question lists the answer spans it should find: a document ID
  and a character range. Sample labels are in
  `data/sample/questions.jsonl`; `questions.md` next to it explains each
  question. A question with no spans has no answer in the corpus.
- **Relevance:** a retrieved chunk is relevant if it overlaps an answer span.
- **Search:** every question is searched against the setup's latest complete run,
  exactly like `raglab query`, and the top k chunks are scored.
- **Checks:** before scoring, the labels are checked against the indexed corpus
  (document exists, span text matches), so labels cannot silently drift.
- **Results:** printed per question, per group and overall, and stored in
  PostgreSQL with the run (`rag.evaluations`).
- **No-answer questions:** excluded from the scores. Only their top score is
  reported, to help choose a no-answer threshold.

## Reading the scores

All scores use the top k results (`--top-k`, default 5). Per-group and overall
values are averages over the answerable questions.

| Score | Range | What it tells you |
|---|---|---|
| [recall](https://en.wikipedia.org/wiki/Precision_and_recall) | 0–1 | Share of a question's answer spans covered by a retrieved chunk. 1.0 means every span was found. |
| [precision](https://en.wikipedia.org/wiki/Precision_and_recall) | 0–1 | Share of the k retrieved chunks that are relevant. It cannot reach 1.0 when fewer than k chunks can be relevant (one span gives at most 1/k). |
| first relevant rank | 1–k, or `-` | Position of the first relevant chunk; `-` means none in the top k. |
| [hit](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)) | 0–1 | Share of questions with at least one relevant chunk in the top k. Ignores position. |
| [mrr](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) | 0–1 | Mean of 1 / (rank of the first relevant chunk), 0 when none was found. Rank 1 gives 1.0, rank 2 gives 0.5, rank 3 gives 0.33. Rewards ranking the first right chunk high. |
| [ndcg](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) | 0–1 | Ranking quality against the ideal ranking (all relevant chunks first, discounted by position). 1.0 means the best possible ranking for that question, even when precision cannot reach 1.0. |

Example for a group of 4 questions whose first relevant chunk was found at ranks
1 and 3, not found, and 2: hit = 3/4 = 0.75 and mrr = (1 + 0.33 + 0 + 0.5) / 4 =
0.46. hit is never lower than mrr. A high hit with a low mrr means the right
chunk is retrieved but ranked low, which is where reranking helps.

Going deeper. The score names in the table link to Wikipedia; these are easier
to start with:

- [Precision and recall at k](https://www.evidentlyai.com/ranking-metrics/precision-recall-at-k),
  [Mean Reciprocal Rank (MRR) explained](https://www.evidentlyai.com/ranking-metrics/mean-reciprocal-rank-mrr)
  and [NDCG explained](https://www.evidentlyai.com/ranking-metrics/ndcg-metric)
  (Evidently AI): short explainers for newcomers. The
  [ranking metrics index](https://www.evidentlyai.com/ranking-metrics) lists the
  whole series.
- [Evaluation measures (information retrieval)](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval))
  (Wikipedia): reference overview of precision, recall, precision at k, MRR, DCG
  and nDCG.

How to use them:

- Compare the same score between setups on the same questions; do not judge a
  single number on its own.
- Use recall and hit to see whether answers are found at all, mrr and ndcg to see
  how high they rank, and precision mainly to compare setups with the same k.
- Chunk size shifts the scores: large chunks overlap spans easily (high recall
  and hit) but fill the top k with long, diluted passages.
- The sample has 25 questions (a few per group), so small differences are noise.
- Questions without an answer (group G) are not scored; compare their mean top
  score with the answerable questions' to choose a score threshold.
