# Comparison table (weighted avg values)

This table lists the weighted average precision, recall, F1, and mean inference latency (seconds) for both ClariQ and AmbigNQ per model and prompt. Avg F1 is the mean of ClariQ and AmbigNQ weighted F1.

**Selection rule:** Rows are sorted by Avg F1 (descending). The top row is suggested as the **baseline** for highest overall weighted F1.

## **Llama Family**

|Method|Prompt|ClariQ P|ClariQ R|ClariQ F1|ClariQ Latency (s)|AmbigNQ P|AmbigNQ R|AmbigNQ F1|AmbigNQ Latency (s)|Avg F1|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Llama-3.3-70B|few_shot|0.86|0.84|0.85|2.9260|0.54|0.57|0.53|2.9871|0.6900|
|Llama-3.3-70B|zero_shot|0.83|0.79|0.81|1.1926|0.53|0.55|0.54|1.1516|0.6750|
|Llama-3.2-3B|zero_shot|0.84|0.88|0.84|0.0601|0.49|0.56|0.47|0.0567|0.6550|
|Llama-3.2-3B|few_shot|0.82|0.52|0.59|0.0798|0.49|0.44|0.42|0.0668|0.5050|
|Llama-3.2-1B|zero_shot|0.82|0.83|0.82|0.0293|0.49|0.56|0.47|0.0317|0.6450|
|Llama-3.2-1B|few_shot|0.77|0.88|0.82|0.0523|0.55|0.59|0.43|0.0353|0.6250|
|Llama-3.1-8B|few_shot|0.86|0.44|0.52|0.0937|0.51|0.48|0.47|0.0718|0.4950|
|Llama-3.1-8B|zero_shot|0.81|0.42|0.50|0.0433|0.52|0.49|0.49|0.0466|0.4950|

## **Gemma Family**

|Method|Prompt|ClariQ P|ClariQ R|ClariQ F1|ClariQ Latency (s)|AmbigNQ P|AmbigNQ R|AmbigNQ F1|AmbigNQ Latency (s)|Avg F1|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Gemma-3-1B|few_shot|0.81|0.86|0.82|0.0247|0.52|0.56|0.50|0.0129|0.6600|
|Gemma-3-1B|zero_shot|0.77|0.87|0.82|0.0076|0.50|0.55|0.49|0.0092|0.6550|
|Gemma-3-27B|zero_shot|0.80|0.77|0.79|0.1203|0.53|0.54|0.53|0.1248|0.6600|
|Gemma-3-27B|few_shot|0.84|0.68|0.73|0.2289|0.51|0.48|0.48|0.1816|0.6050|
|Gemma-3-4B|few_shot|0.77|0.87|0.82|0.0685|0.50|0.58|0.43|0.0724|0.6250|
|Gemma-3-4B|zero_shot|0.77|0.88|0.82|0.0670|0.34|0.59|0.43|0.0399|0.6250|

## **Phi Family**

|Method|Prompt|ClariQ P|ClariQ R|ClariQ F1|ClariQ Latency (s)|AmbigNQ P|AmbigNQ R|AmbigNQ F1|AmbigNQ Latency (s)|Avg F1|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Phi-3|few_shot|0.83|0.76|0.78|0.0688|0.53|0.56|0.51|0.0474|0.6450|
|Phi-3|zero_shot|0.77|0.88|0.82|0.0782|0.34|0.59|0.43|0.0304|0.6250|

## **OpenAI / GPT-4.1 Family**

|Method|Prompt|ClariQ P|ClariQ R|ClariQ F1|ClariQ Latency (s)|AmbigNQ P|AmbigNQ R|AmbigNQ F1|AmbigNQ Latency (s)|Avg F1|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|GPT-4.1|zero_shot|0.84|0.68|0.73|N/A|0.61|0.54|0.53|N/A|0.6300|
|GPT-4.1|few_shot|0.85|0.49|0.56|N/A|0.56|0.48|0.44|N/A|0.5000|
|GPT-4.1-mini|zero_shot|0.85|0.71|0.76|N/A|0.58|0.51|0.49|N/A|0.6250|
|GPT-4.1-mini|few_shot|0.86|0.63|0.69|N/A|0.56|0.48|0.44|N/A|0.5650|
|GPT-4.1-nano|zero_shot|0.81|0.58|0.65|N/A|0.54|0.54|0.54|N/A|0.5950|
|GPT-4.1-nano|few_shot|0.82|0.32|0.37|N/A|0.54|0.45|0.38|N/A|0.3750|


### Notes
- Values are taken from the `weighted avg` row in each log's classification report and the `Mean:` line from inference time stats. Latency is in seconds.
- `Avg F1` is the simple mean of the ClariQ F1 and AmbigNQ F1 for a row.
- The top method (**Llama-3.3-70B few_shot**) is recommended as a **performance-first** baseline. If low latency is priority, `Gemma-3-1B few_shot` is a good latency-aware alternative (Avg F1 0.66 and mean latency ≈ 0.02s).

---

## Pre-trained Language Model (PLM) Classifier Results

The table below shows results for fine-tuned PLM classifiers trained on LLM-generated embeddings. These classifiers are trained on embeddings from LLM predictions and provide a lightweight alternative to direct LLM inference.

### **Llama Embeddings**

|PLM Model|Embedding Source|ClariQ P|ClariQ R|ClariQ F1|ClariQ Latency (s)|AmbigNQ P|AmbigNQ R|AmbigNQ F1|AmbigNQ Latency (s)|Avg F1|Avg Latency(s)|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|RoBERTa|Llama-3.1-8B|0.81|0.82|0.81|0.0043|0.48|0.46|0.47|0.0040|0.6400|0.0042|
|DistilBERT|Llama-3.3-70B|0.83|0.76|0.79|0.0020|0.50|0.49|0.49|0.0020|0.6400|0.0020|
|DistilBERT|Llama-3.1-8B|0.83|0.78|0.80|0.0025|0.52|0.47|0.45|0.0025|0.6250|0.0025|
|RoBERTa|Llama-3.3-70B|0.82|0.53|0.60|0.0035|0.52|0.54|0.52|0.0032|0.5600|0.0034|

### **GPT-4.1 Embeddings**

|PLM Model|Embedding Source|ClariQ P|ClariQ R|ClariQ F1|ClariQ Latency (s)|AmbigNQ P|AmbigNQ R|AmbigNQ F1|AmbigNQ Latency (s)|Avg F1|Avg Latency(s)|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|RoBERTa|GPT-4.1|0.82|0.86|0.83|0.0017|0.51|0.57|0.48|0.0016|0.6550|0.0017|
|DistilBERT|GPT-4.1 mini|0.82|0.60|0.67|0.0019|0.51|0.49|0.49|0.0015|0.5800|0.0017|
|DistilBERT|GPT-4.1 nano|0.81|0.63|0.69|0.0019|0.52|0.45|0.41|0.0017|0.5500|0.0018|
|DistilBERT|GPT-4.1|0.82|0.59|0.66|0.0018|0.54|0.47|0.43|0.0017|0.5450|0.0017|
|RoBERTa|GPT-4.1 nano|0.79|0.45|0.54|0.0017|0.51|0.50|0.50|0.0016|0.5200|0.0017|
|RoBERTa|GPT-4.1 mini|0.83|0.36|0.42|0.0029|0.54|0.45|0.39|0.0019|0.4050|0.0024|

### **Gemma Embeddings**

|PLM Model|Embedding Source|ClariQ P|ClariQ R|ClariQ F1|ClariQ Latency (s)|AmbigNQ P|AmbigNQ R|AmbigNQ F1|AmbigNQ Latency (s)|Avg F1|Avg Latency(s)|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|DistilBERT|Gemma-3-27B|0.82|0.56|0.64|0.0020|0.57|0.44|0.34|0.0019|0.4900|0.0020|
|RoBERTa|Gemma-3-27B|0.83|0.37|0.43|0.0033|0.55|0.45|0.38|0.0031|0.4050|0.0032|

### Notes
- PLM classifiers are trained on embeddings generated by the specified LLM (Embedding Source).
- Values are taken from the `weighted avg` row in each PLM classification report.
- Latency values are per-sample average inference times.
- **New PLM rows (RoBERTa + GPT-4.1 / mini / nano)** were added to the GPT-4.1 embeddings group. The RoBERTa classifier trained on full GPT-4.1 embeddings (`RoBERTa + GPT-4.1`) now achieves the highest Avg F1 among PLMs (0.6550), slightly above the prior top Avg F1 entries (0.64).
- PLM classifier best performers (post-update):
	- **RoBERTa + GPT-4.1** — Avg F1 0.6550; Avg Latency ≈ 0.0017s.
	- **RoBERTa + Llama-3.1-8B** and **DistilBERT + Llama-3.3-70B** — Avg F1 0.64.
- PLM classifiers offer large speedups compared to direct LLM inference, but the exact factor varies greatly depending on model pairings and measured latencies. Here are a few examples to illustrate:
	- **DistilBERT + Llama-3.3-70B** latency ≈ 0.0020s vs **Llama-3.3-70B (few_shot)** ≈ 2.9260s → ≈ 1460x speedup.
	- **RoBERTa + Llama-3.1-8B** latency ≈ 0.0043s vs **Llama-3.1-8B (few_shot)** ≈ 0.0937s → ≈ 22x speedup.
	- **DistilBERT + Gemma-3-27B** latency ≈ 0.0020s vs **Gemma-3-1B (few_shot)** ≈ 0.0247s → ≈ 12x speedup.
	- These examples demonstrate that observed speedups can range from the low tens to over a thousand times faster for PLMs, depending largely on the LLM baseline used.
- **Latency caveats:** Some LLM entries report `N/A` for latency (OpenAI/GPT-4.1 rows). When present, PLM latency is typically reported as per-sample average inference time measured locally.
- **Interpretation & recommendations:**
	- For maximum accuracy (no latency constraints): **Llama-3.3-70B (few_shot)** remains the top LLM choice.
	- For production settings where latency matters and performance needs to be close to LLMs: use a **PLM classifier trained on high-quality embeddings**, e.g., **RoBERTa + GPT-4.1** for the highest PLM Avg F1 or **DistilBERT + Llama-3.3-70B** for the best PLM latency / performance balance.
	- For very low-latency scenarios with minimal compute cost: **Gemma-3-1B (few_shot)** for direct LLM inference is a latency-aware LLM choice; alternatively, PLMs trained on Gemma embeddings will further reduce inference latency.
- **Practical note:** If embeddings are precomputed (offline) or cached, PLM classifiers enable very fast, cheap inference suitable for scalable production deployments while preserving reasonable classification performance.