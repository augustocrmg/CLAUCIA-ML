"""Agrega os resultados dos folds (resultados/<size>_<adaptation>/fold*/).

Para cada configuração:
  - média ± desvio-padrão por fold (teste e E1);
  - predições de teste empilhadas dos 5 folds (cada original aparece uma vez):
    sensibilidade, especificidade, precisão e F1 por classe, com IC 95% por
    bootstrap estratificado;
  - calibração: ECE (15 bins), Brier e NLL, antes e depois de temperature
    scaling (T ajustado por fold na validação daquele fold... aqui, por falta
    das probabilidades de validação, T é ajustado por validação cruzada nas
    predições de teste dos outros folds).
Saída: resultados/resumo.md e resultados/resumo.json
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

ROOT = Path(os.environ.get("CBEB_RESULTS", Path(__file__).resolve().parent / "resultados"))
CLASSES = ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4", "Grade 5", "Normal"]
RNG = np.random.default_rng(42)
N_BOOT = 2000


def load(pred_csv):
    df = pd.read_csv(pred_csv)
    p = df[[f"prob_{c}" for c in CLASSES]].to_numpy()
    y = df.classe_real.map(CLASSES.index).to_numpy()
    return y, p


def ece(y, p, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return total


def brier(y, p):
    return float(((p - np.eye(len(CLASSES))[y]) ** 2).sum(1).mean())


def nll(y, p):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean())


def temperature(p, T):
    logits = np.log(np.clip(p, 1e-12, None)) / T
    e = np.exp(logits - logits.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def fit_T(y, p):
    return minimize_scalar(lambda T: nll(y, temperature(p, T)), bounds=(0.05, 10), method="bounded").x


def per_class(y, p):
    pred = p.argmax(1)
    rows = {}
    for c in range(len(CLASSES)):
        tp = ((pred == c) & (y == c)).sum(); fn = ((pred != c) & (y == c)).sum()
        fp = ((pred == c) & (y != c)).sum(); tn = ((pred != c) & (y != c)).sum()
        sens = tp / (tp + fn) if tp + fn else np.nan
        prec = tp / (tp + fp) if tp + fp else np.nan
        rows[CLASSES[c]] = dict(sens=sens, spec=tn / (tn + fp), prec=prec,
                                f1=2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan, n=int(tp + fn))
    return rows


def bootstrap_ci(y, p, fn):
    idx_by_class = [np.where(y == c)[0] for c in range(len(CLASSES))]
    vals = []
    for _ in range(N_BOOT):
        idx = np.concatenate([RNG.choice(ix, len(ix)) for ix in idx_by_class if len(ix)])
        vals.append(fn(y[idx], p[idx]))
    return np.nanpercentile(np.array(vals, dtype=float), [2.5, 97.5], axis=0)


def summarize(cfg_dir):
    folds = sorted(cfg_dir.glob("fold*/summary.json"))
    if not folds:
        return None
    s = [json.loads(f.read_text()) for f in folds]
    out = {"config": cfg_dir.name, "n_folds": len(s),
           "params_total": s[0]["params_total"], "params_trainable": s[0]["params_trainable"],
           "train_minutes": float(np.mean([x["train_minutes"] for x in s])),
           "best_epoch": [x["best_epoch"] for x in s]}
    for split in ["test", "e1"]:
        for k in ["accuracy", "f1", "recall", "auc_roc"]:
            v = [x[split][k] for x in s]
            out[f"{split}_{k}_mean"], out[f"{split}_{k}_std"] = float(np.mean(v)), float(np.std(v, ddof=1) if len(v) > 1 else 0)

    # Predições de teste empilhadas: cada original do núcleo aparece uma vez.
    ys, ps, fold_id = [], [], []
    for i, f in enumerate(folds):
        y, p = load(f.parent / "predictions_test.csv")
        ys.append(y); ps.append(p); fold_id.append(np.full(len(y), i))
    y, p, fold_id = np.concatenate(ys), np.concatenate(ps), np.concatenate(fold_id)
    out["pooled_n"] = int(len(y))
    out["pooled_accuracy"] = float(accuracy_score(y, p.argmax(1)))
    out["pooled_f1_macro"] = float(f1_score(y, p.argmax(1), average="macro"))
    out["pooled_bal_acc"] = float(recall_score(y, p.argmax(1), average="macro"))
    out["pooled_auc_roc"] = float(roc_auc_score(y, p, multi_class="ovr", average="macro"))
    ci = bootstrap_ci(y, p, lambda yy, pp: f1_score(yy, pp.argmax(1), average="macro"))
    out["pooled_f1_macro_ci"] = ci.tolist()

    pc = per_class(y, p)
    keys = ["sens", "spec"]
    ci_pc = bootstrap_ci(y, p, lambda yy, pp: [per_class(yy, pp)[c][k] for c in CLASSES for k in keys])
    for j, (c, k) in enumerate([(c, k) for c in CLASSES for k in keys]):
        pc[c][f"{k}_ci"] = [float(ci_pc[0][j]), float(ci_pc[1][j])]
    out["per_class"] = pc

    # Calibração: T ajustado sem olhar o próprio fold (leave-one-fold-out).
    p_cal = np.empty_like(p)
    Ts = []
    for i in np.unique(fold_id):
        T = fit_T(y[fold_id != i], p[fold_id != i]); Ts.append(T)
        p_cal[fold_id == i] = temperature(p[fold_id == i], T)
    out["calibration"] = {"ece": ece(y, p), "brier": brier(y, p), "nll": nll(y, p),
                          "T_mean": float(np.mean(Ts)),
                          "ece_ts": ece(y, p_cal), "brier_ts": brier(y, p_cal), "nll_ts": nll(y, p_cal),
                          "mean_conf_errors": float(p.max(1)[p.argmax(1) != y].mean())}

    # E1 empilhado por média das probabilidades dos 5 modelos (ensemble dos folds).
    e1 = [load(f.parent / "predictions_e1.csv") for f in folds]
    ye1, pe1 = e1[0][0], np.mean([q for _, q in e1], axis=0)
    present = sorted(set(ye1))
    out["e1_ensemble"] = {"accuracy": float(accuracy_score(ye1, pe1.argmax(1))),
                          "f1_macro_present": float(f1_score(ye1, pe1.argmax(1), labels=present, average="macro")),
                          "per_class_sens": {CLASSES[c]: float((pe1.argmax(1)[ye1 == c] == c).mean()) for c in present},
                          "pred_distribution": {CLASSES[c]: int((pe1.argmax(1) == c).sum()) for c in range(len(CLASSES))}}
    return out


def main():
    results = [r for d in sorted(ROOT.iterdir()) if d.is_dir() and (r := summarize(d))]
    (ROOT / "resumo.json").write_text(json.dumps(results, indent=2, default=float))
    lines = ["| Config | folds | Teste acc | Teste F1m | F1m empilhado [IC95] | AUC | E1 acc | E1 F1m | ECE → TS | min/fold |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        c = r["calibration"]
        lines.append(f"| {r['config']} | {r['n_folds']} | {r['test_accuracy_mean']:.3f}±{r['test_accuracy_std']:.3f} | "
                     f"{r['test_f1_mean']:.3f}±{r['test_f1_std']:.3f} | {r['pooled_f1_macro']:.3f} "
                     f"[{r['pooled_f1_macro_ci'][0]:.3f}, {r['pooled_f1_macro_ci'][1]:.3f}] | {r['pooled_auc_roc']:.3f} | "
                     f"{r['e1_accuracy_mean']:.3f}±{r['e1_accuracy_std']:.3f} | {r['e1_f1_mean']:.3f} | "
                     f"{c['ece']:.3f} → {c['ece_ts']:.3f} | {r['train_minutes']:.1f} |")
    for r in results:
        lines += ["", f"### {r['config']}: por classe (teste empilhado, n={r['pooled_n']})",
                  "| Classe | n | Sensibilidade [IC95] | Especificidade [IC95] | Precisão | F1 |", "|---|---|---|---|---|---|"]
        for cls, v in r["per_class"].items():
            lines.append(f"| {cls} | {v['n']} | {v['sens']:.3f} [{v['sens_ci'][0]:.2f}, {v['sens_ci'][1]:.2f}] | "
                         f"{v['spec']:.3f} [{v['spec_ci'][0]:.2f}, {v['spec_ci'][1]:.2f}] | {v['prec']:.3f} | {v['f1']:.3f} |")
        lines.append(f"\nE1 (ensemble dos folds): {json.dumps(r['e1_ensemble'], default=float)}")
    (ROOT / "resumo.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
