---
version: 0.1.0
created: 2026-03-09
---

# authentication-scanner

Frozen DINOv2 flags paintings by the wrong artist (p=2e-11) but can't tell Rembrandt from his circle (63.7% ceiling)

> **Status:** complete, March 2026. No further experiments planned. Reusable: the open-collection fetch pipeline, entropy-weighted tile aggregation, the probe harness and the decisions log.

| Task | Result | Verdict |
|------|--------|---------|
| Autograph vs pupil/other Dutch | p=2e-11, p=3e-10 | Strong: reliably detects different artist |
| Autograph vs circle | p=0.80 (unsupervised), 63.7% (supervised) | Weak: can't separate master from skilled imitator |
| Hard negatives (top 25% hardest circle) | 55.1% balanced accuracy, p=0.228 | Dead: no edge on commercially relevant cases |

## The question

Can frozen DINOv2 embeddings screen open museum collections and flag misattributed paintings for expert review? Rembrandt is the test case: 1,311 paintings across 4 attribution levels (autograph, circle, pupil, dutch_other). The case that matters commercially is master vs skilled imitator, which is the circle group (works catalogued as Rembrandt's circle, workshop, followers or "style of").

## What we found

DINOv2 frozen embeddings can detect "wrong artist" misattributions (p=2e-11) but cannot distinguish master from skilled imitator, the commercially valuable question. After an exhaustive 3-tier search the ceiling is 63.7% balanced accuracy (entropy-weighted ViT-B/14 tiles with an SVM RBF probe).

The unsupervised p-values in the table come from the 108-painting Rijksmuseum and Met set (v1, ViT-B/14, Mann-Whitney). Supervised accuracies are 10-fold stratified CV on the 711 autograph and circle paintings, with balanced class weights. At 1,311 paintings the unsupervised circle difference does reach significance (p≈0.000) but no supervised probe gets past 63.7%.

### Exhaustive search: nothing left to try with frozen embeddings

| Tier | What we tried | Best result |
|------|---------------|-------------|
| **Tier 1: Frozen features** | ViT-B mean (59.0%), ViT-L (59.5%), entropy (63.7%), CLIP (62.9%), concat (62.3%), per-tile voting (60.2%), fine-tune (60.0%) | 63.7% |
| **Tier 2: Diagnostics** | Confounder audit, robustness, calibration, hard-negatives, domain-shift, multimodal | Signal real but marginal; no step change |
| **Tier 3: LoRA + transfer** | LoRA Rembrandt (60.9%), LoRA leave-artist-out (crashed), curriculum transfer (59.0%) | All underperform frozen 63.7% |

Every configuration, with its statistics, is in [docs/scan-results.md](docs/scan-results.md).

## How

`scan.py` is a CLI pipeline that fetches paintings from 6 museum APIs + Wikidata SPARQL, computes DINOv2 vision embeddings and runs unsupervised and supervised classification probes. It has 78 tests, runs on CUDA or MPS and comes with GCE T4 deployment scripts. Each stage caches its output (metadata, then images, then embeddings) so re-runs skip finished work.

The work ran from 26 February to 9 March 2026 (about 12 days), in phases:

1. **Prototype** (Rijksmuseum, 70 paintings): 82.9% KNN against a 72.9% baseline, pupils p=0.001.
2. **Met expansion** (108 paintings, 18 circle): circle p=0.805, still inseparable. High-res tiles with std features (v2) pushed every similarity towards 0.90 and lost the pupil signal.
3. **Wikidata SPARQL expansion** (1,311 paintings, 149 circle): the first balanced supervised probe reached 59.0% (p=0.003). The earlier 72.3% on 47 paintings had been overfit.
4. **Tier 1, frozen features:** seven configurations. Entropy-weighted tiles came out best at 63.7%.
5. **Tier 2, diagnostics:** confounder audit, robustness test, calibration, hard negatives, domain shift and a metadata probe. The signal is real but marginal.
6. **Tier 3, LoRA and multi-artist transfer** on a GCE T4 (Rubens, Cranach, Van Dyck, Titian, Hals and Rembrandt): every variant that finished came in under 63.7%.

| File | What |
|------|------|
| [docs/decisions.md](docs/decisions.md) | 33-entry chronological decisions log |
| [docs/scan-results.md](docs/scan-results.md) | Full experimental results with statistics |
| [docs/prototype-results.md](docs/prototype-results.md) | Prototype on the Rijksmuseum Rembrandt corpus |
| [docs/adjacent-angles.md](docs/adjacent-angles.md) | Assessment of 10 candidate extensions |
| [docs/research.md](docs/research.md) | Market opportunity research |
| [docs/execution.md](docs/execution.md) | Competitor deep-dive, pricing benchmarks, risk analysis |
| [docs/graduation-tracking.md](docs/graduation-tracking.md) | Gate documentation |
| [SPEC.md](SPEC.md) | Retroactive build contract |
| [prototype.ipynb](prototype.ipynb) | Phase 1 prototype notebook |

## Reproduce

Python 3.10 or later. `scan.py` picks CUDA, MPS or CPU automatically and downloads the DINOv2 weights through `torch.hub` on first use. The data APIs are free and need no keys.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ml,dev]" imagehash

python scan.py                                 # fetch, download, dedup, embed (ViT-B/14 mean), analyse
python scan.py --entropy                       # entropy-weighted embeddings (the best configuration)
python scan.py --probe --entropy --legacy-cv   # autograph vs circle probe, same CV as the 63.7% figure
python scan.py --calibration                   # one of the Tier 2 diagnostics
pytest tests/ -x -q
```

The other Tier 2 diagnostics are `--confounder-audit`, `--robustness-test`, `--hard-negatives`, `--domain-shift` and `--multimodal`. All of them read the entropy embeddings.

Tier 3 needs a GPU. [deploy-gpu.sh](deploy-gpu.sh) creates a GCE T4 VM (36-hour hard cap) that runs the experiments listed in [startup-gpu.sh](startup-gpu.sh), uploads the results to a GCS bucket and deletes itself. Set `GCP_PROJECT` and `GCS_BUCKET` first. The Tier 3 flags are `--corpus transfer` (fetch and embed the six-artist corpus), `--lora`, `--lora-transfer`, `--lora-curriculum` and `--transfer-probe`.

A rerun won't match the published numbers exactly:

- **Data drift.** Wikidata keeps changing, and the default perceptual dedup (`--dedup-threshold 10`) was added after the corpus behind these numbers was embedded. Expect a count other than 1,311.
- **Cross-validation.** `--probe` defaults to nested CV (outer 10-fold, inner 5-fold), which came after the 63.7% run. `--legacy-cv` matches the published method. No nested-CV result for the headline probe is recorded in this repo.
- **Sources.** The data came from Rijksmuseum, Met Open Access and Wikidata SPARQL (images from Wikimedia Commons). The NGA, CMA and AIC clients returned nothing in March 2026 (NGA gave a 404; CMA and AIC had no Rembrandt matches).
- **Dependencies.** The dedup step imports `imagehash`, which `pyproject.toml` doesn't declare, hence the extra package in the install line.

Time and cost of the original run:

- Compute: <$100 (GCE Spot/on-demand T4, ~10 GPU-hours total)
- Data: $0 (all museum APIs are CC0, no keys)
- Time: ~12 days part-time

## Limits and prior art

- The headline numbers are Rembrandt only. The five other artists were used for transfer learning in Tier 3.
- Labels come from Wikidata attribution qualifiers and museum catalogue text, so some are noisy ([SPEC.md](SPEC.md) accepts that trade-off).
- The Rembrandt corpus folds workshop into circle, so workshop vs autograph separability was never measured.
- Experiment A (frozen transfer probe) never ran because its embeddings were lost with a crashed VM. Experiment C crashed after its Cranach fold (53.4%).

### Why the competitors can do it (and we can't with this approach)

Art Recognition and Hephaestus train **custom supervised models per artist** on brushstroke-level features instead of using frozen foundation model embeddings. Our approach traded training time for signal strength. Validation was fast but the ceiling is real.

Training a custom model would take months of work, significant GPU compute and tile-level brushstroke annotation. It would also mean competing head-to-head with companies that have a 7-year head start, art-world relationships and (for Hephaestus) an insurance product. Competitor detail and pricing are in [docs/execution.md](docs/execution.md).

## Open questions and reusable parts

Technical exploration is exhausted: frozen vision embeddings can't separate master from imitator at commercially useful accuracy. The research, the pipeline and the 33-entry decisions log still have reference value for future vision projects.

Reusable:

- **Open-collection fetchers** for Rijksmuseum, Met Open Access and Wikidata SPARQL (P170 creator with the P1774–P1780 attribution qualifiers), behind a three-tier cache. Swapping the artist QID retargets the Wikidata query.
- **Entropy-weighted tile aggregation**, which weights DINOv2 tiles by patch-token variance (z-score, then softmax) so faces, hands and brushwork count for more than flat background.
- **Probe harness:** PCA, then Logistic, SVM RBF or MLP with balanced class weights, nested or legacy CV and permutation tests.
- **GCE deploy scripts** for one-shot GPU runs that upload results and delete the VM.

Open:

- **Provenance and metadata.** A metadata probe (dimensions, material, dates) added 0.7pp on top of the embeddings. Provenance itself was never tested.
- **Diagnostic region crops** (hands, eyes, ears, fabric folds), the areas art historians zoom into.
- **Craquelure, canvas and pigment** signal from native-resolution images.
- **Workshop separability** on the six-artist corpus, which labels workshop separately (468 paintings).
- **Cross-tradition transfer** (Old Master drawings, prints, Chinese painting), sketched as Tier 4 in [docs/scan-results.md](docs/scan-results.md).

Issues and replications are welcome.

## Attribution

For those who care about such things, the majority of the analysis and code in this repo was produced by Claude Opus 4.6 (Anthropic), with David Fitzgerald guiding Claude, providing insights, direction and critical review throughout the process.

## License

MIT. See [LICENSE](LICENSE).
