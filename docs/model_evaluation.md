# Model evaluation (Vultr Serverless Inference)

**Date:** 14 November 2026 · **Script:** `python -m app.compare_models`

We compared four chat models available on Vultr Serverless Inference on the same real student
question, using BacBuddy's full RAG pipeline (query expansion → BM25 retrieval → grounded answer).

**Test question:** *"Cum se punctează studiul monotoniei unei funcții la Bacalaureat?"*
(How is the study of a function's monotonicity graded at the Baccalaureate?)

| Model | Latency | Sources cited | Grounded? | Notes |
|---|---|---|---|---|
| `glm-5.3` | 37.6 s | 3 | ✅ | Good Romanian, but far too slow for an interactive app |
| `deepseek-v4.1-flash-1017` | **3.8 s** | 5 | ✅ | Fastest, well structured, clearly separates what sources cover vs. not |
| `qwen3.8-27b` | 5.1 s | 6 | ✅ | Correct but short, cited all sources in one sentence |
| `mimo-v2.6-pro` | 6.2 s | 5 | ✅ | Very good structure, slower |

## Decision

We use **`deepseek-v4.1-flash-1017`**: best balance of speed, answer quality and citation discipline.

## What the evaluation revealed (and what we fixed)

1. **No hallucinations.** When the retrieved sources didn't contain the grading points, all four
   models said so instead of inventing numbers, which confirms the grounding prompt works.
2. **Retrieval gap:** grading schemes (bareme) don't use the word "monotonie"; they say
   "f'(x) ≥ 0, deci f este crescătoare … 3p". Keyword search only found the curriculum.
   **Fix:** LLM query expansion (adds terms like *crescătoare, descrescătoare, semnul derivatei*)
   plus guaranteed barem results for grading questions.
3. **Wrong subject:** a Math question pulled Computer Science grading schemes.
   **Fix:** automatic subject detection from the question (e.g. *funcție, derivată* → Math;
   *algoritm, subprogram, C++* → Computer Science).
4. **Partial answers:** the prompt now asks the model to answer the part the sources cover and
   state clearly what the official materials don't specify.