# CLI

The CLI operates on a small set of setups and ingestion templates predefined in
code. A setup fixes chunking, embedding, retrieval, and Qdrant indexing choices;
a template converts one input format into normalized documents.

```bash
raglab setups
```

Lists supported setup IDs and their main settings. Use it to discover valid
values for `--setup`.

```bash
raglab ingest data/squad --template squad-contexts --setup naive-500-100
```

Normalizes input documents, then chunks and embeds them before writing vectors
and setup metadata to Qdrant. Store the setup ID, resolved parameters, template
ID, and source/corpus version with every ingestion run.

Initial code-defined templates can include `msmarco-passages`, `squad-contexts`,
`markdown-directory`, and `jsonl-documents`. For SQuAD, index unique contexts
and retain questions/answer spans for evaluation rather than treating questions
as source documents.

```bash
raglab query "What is late chunking?" --setup naive-500-100
```

Embeds the query and searches only vectors indexed for the selected setup.

```bash
raglab status
```

Shows each setup's indexed document/vector counts, target collection, last
ingestion time, and readiness state.

```bash
raglab reset --setup naive-500-100
```

Removes only vectors and index metadata for the selected setup, allowing a clean
re-index. It should request confirmation; `--yes` may bypass it for automation.

