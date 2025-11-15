# Baseline comparison table (weighted avg values)

This table lists the weighted average precision, recall, F1, and mean inference latency (seconds) for both ClariQ and AmbigNQ per model and prompt. Avg F1 is the mean of ClariQ and AmbigNQ weighted F1.

**Selection rule:** Rows are sorted by Avg F1 (descending). The top row is suggested as the **baseline** for highest overall weighted F1.

|Method|Prompt|ClariQ P|ClariQ R|ClariQ F1|ClariQ Latency (s)|AmbigNQ P|AmbigNQ R|AmbigNQ F1|AmbigNQ Latency (s)|Avg F1|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|**Llama-3.3-70B**|few_shot|0.86|0.84|0.85|2.9260|0.54|0.57|0.53|2.9871|0.6900|
|Llama-3.3-70B|zero_shot|0.83|0.79|0.81|1.1926|0.53|0.55|0.54|1.1516|0.6750|
|Gemma-3-1B|few_shot|0.81|0.86|0.82|0.0247|0.52|0.56|0.50|0.0129|0.6600|
|Gemma-3-27B|zero_shot|0.80|0.77|0.79|0.1203|0.53|0.54|0.53|0.1248|0.6600|
|Llama-3.2-3B|zero_shot|0.84|0.88|0.84|0.0601|0.49|0.56|0.47|0.0567|0.6550|
|Gemma-3-1B|zero_shot|0.77|0.87|0.82|0.0076|0.50|0.55|0.49|0.0092|0.6550|
|Phi-3|few_shot|0.83|0.76|0.78|0.0688|0.53|0.56|0.51|0.0474|0.6450|
|Llama-3.2-1B|zero_shot|0.82|0.83|0.82|0.0293|0.49|0.56|0.47|0.0317|0.6450|
|Gemma-3-4B|few_shot|0.77|0.87|0.82|0.0685|0.50|0.58|0.43|0.0724|0.6250|
|Gemma-3-4B|zero_shot|0.77|0.88|0.82|0.0670|0.34|0.59|0.43|0.0399|0.6250|
|Phi-3|zero_shot|0.77|0.88|0.82|0.0782|0.34|0.59|0.43|0.0304|0.6250|
|Llama-3.2-1B|few_shot|0.77|0.88|0.82|0.0523|0.55|0.59|0.43|0.0353|0.6250|
|Gemma-3-27B|few_shot|0.84|0.68|0.73|0.2289|0.51|0.48|0.48|0.1816|0.6050|
|Llama-3.2-3B|few_shot|0.82|0.52|0.59|0.0798|0.49|0.44|0.42|0.0668|0.5050|
|Llama-3.1-8B|few_shot|0.86|0.44|0.52|0.0937|0.51|0.48|0.47|0.0718|0.4950|
|Llama-3.1-8B|zero_shot|0.81|0.42|0.50|0.0433|0.52|0.49|0.49|0.0466|0.4950|


### Notes
- Values are taken from the `weighted avg` row in each log's classification report and the `Mean:` line from inference time stats. Latency is in seconds.
- `Avg F1` is the simple mean of the ClariQ F1 and AmbigNQ F1 for a row.
- The top method (**Llama-3.3-70B few_shot**) is recommended as a **performance-first** baseline. If low latency is priority, `Gemma-3-1B few_shot` is a good latency-aware alternative (Avg F1 0.66 and mean latency ≈ 0.02s).

---

## Recall-prioritized baseline (emphasizes recall values)

You specified that recall should be emphasized (you prefer to flag a query as ambiguous rather than mistakenly call it unambiguous). The table below prioritizes **Avg Recall** (mean of ClariQ recall and AmbigNQ recall) as the sorting metric — top row is the recommended recall-first baseline. Ties are broken by Avg F1 and then by Avg Latency (lower better).

|Method|Prompt|ClariQ P|ClariQ R|ClariQ F1|ClariQ Latency (s)|AmbigNQ P|AmbigNQ R|AmbigNQ F1|AmbigNQ Latency (s)|Avg Recall|Avg F1|Avg Latency(s)|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|Gemma-3-4B|zero_shot|0.77|0.88|0.82|0.0670|0.34|0.59|0.43|0.0399|0.7350|0.6250|0.0535|
|Phi-3|zero_shot|0.77|0.88|0.82|0.0782|0.34|0.59|0.43|0.0304|0.7350|0.6250|0.0543|
|Llama-3.2-1B|few_shot|0.77|0.88|0.82|0.0523|0.55|0.59|0.43|0.0353|0.7350|0.6250|0.0438|
|Gemma-3-4B|few_shot|0.77|0.87|0.82|0.0685|0.50|0.58|0.43|0.0724|0.7250|0.6250|0.0705|
|Llama-3.2-3B|zero_shot|0.84|0.88|0.84|0.0601|0.49|0.56|0.47|0.0567|0.7200|0.6550|0.0584|
|Gemma-3-1B|few_shot|0.81|0.86|0.82|0.0247|0.52|0.56|0.50|0.0129|0.7100|0.6600|0.0188|
|Gemma-3-1B|zero_shot|0.77|0.87|0.82|0.0076|0.50|0.55|0.49|0.0092|0.7100|0.6550|0.0084|
|Llama-3.3-70B|few_shot|0.86|0.84|0.85|2.9260|0.54|0.57|0.53|2.9871|0.7050|0.6900|2.9566|
|Llama-3.3-70B|zero_shot|0.83|0.79|0.81|1.1926|0.53|0.55|0.54|1.1516|0.6700|0.6750|1.1721|
|Llama-3.2-1B|zero_shot|0.82|0.83|0.82|0.0293|0.49|0.56|0.47|0.0317|0.6950|0.6450|0.0305|
|Phi-3|few_shot|0.83|0.76|0.78|0.0688|0.53|0.56|0.51|0.0474|0.6600|0.6450|0.0581|
|Gemma-3-27B|zero_shot|0.80|0.77|0.79|0.1203|0.53|0.54|0.53|0.1248|0.6550|0.6600|0.1226|
|Gemma-3-27B|few_shot|0.84|0.68|0.73|0.2289|0.51|0.48|0.48|0.1816|0.5800|0.6050|0.2052|
|Llama-3.2-3B|few_shot|0.82|0.52|0.59|0.0798|0.49|0.44|0.42|0.0668|0.4800|0.5050|0.0733|
|Llama-3.1-8B|zero_shot|0.81|0.42|0.50|0.0433|0.52|0.49|0.49|0.0466|0.4550|0.4950|0.0449|
|Llama-3.1-8B|few_shot|0.86|0.44|0.52|0.0937|0.51|0.48|0.47|0.0718|0.4600|0.4950|0.0828|

### Recall-first Baseline recommendation
Rounded numbers: the highest average recall is **0.735** (Gemma-3-4B zero_shot, Phi-3 zero_shot, and Llama-3.2-1B few_shot tie). 
Recommendation: choose **Gemma-3-4B zero_shot** as a recall-first baseline because it has the highest Avg Recall (0.735) and a reasonable Avg F1 (0.625) and moderate Avg Latency (0.0535s). If latency is more critical, prefer **Llama-3.2-1B (few_shot)** (Avg Recall 0.735, Avg F1 0.625, Avg Latency 0.0438s).

### Summary
- Reordering by Average Recall emphasizes the models that tend to flag ambiguous queries rather than label them as unambiguous.
- This helps reduce false negatives for the ambiguous class; however, it can increase false positives (more queries flagged ambiguous).
- If you'd like a custom scoring function that balances recall vs F1 vs latency, tell me your preferred weights (e.g., recall weight=0.7, F1 weight=0.2, latency penalty weight=0.1) and I will compute and rank models accordingly.