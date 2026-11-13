"""Métricas A5 (calibração, sensibilidade/especificidade por classe, erro ordinal) para os
modelos treinados pelo professor, sobre as predições do teste original (test_predictions.csv).

  - teste completo (n = 1050, inclui aumentos) e subconjunto limpo (prof_test_originals.csv, clean);
  - temperature scaling com T ajustado na validação do professor (resultados_prof_val_<size>.csv);
  - QWK e acurácia ±1 grau com a ordem Normal < G0 < ... < G5.
Saída: resultados_prof_a5.md / .json
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, recall_score, roc_auc_score

import aggregate_results as ag

HERE = Path(__file__).resolve().parent
RES = HERE / "results" / "divisao_original"
PROF = Path(os.environ.get("CLAUCIA_PROF_RESULTS", RES))  # predições do treino de referência (notebook)
EXP = "lora_qv_r8_warmup_v3_stable_large"
CLASSES = ag.CLASSES
ORD = np.array([1, 2, 3, 4, 5, 6, 0])  # posição ordinal de cada classe (Normal = 0)


def load_test(size):
    df = pd.read_csv(PROF / size / EXP / "test_predictions.csv")
    p = df[[c for c in df.columns if c.startswith("prob_")]].to_numpy()
    y = df.classe_real.str.replace("Normal(Healthy skin)", "Normal", regex=False).map(CLASSES.index).to_numpy()
    return df.arquivo, y, p


def load_val(size):
    f = RES / f"resultados_prof_val_{size}.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    p = df[[c for c in df.columns if c.startswith("prob_")]].to_numpy()
    y = df.classe_real.str.replace("Normal(Healthy skin)", "Normal", regex=False).map(CLASSES.index).to_numpy()
    return y, p


def ordinal(y, p):
    pred = p.argmax(1)
    top2 = np.argsort(-p, 1)[:, :2]
    return {"qwk": float(cohen_kappa_score(ORD[y], ORD[pred], weights="quadratic")),
            "within1": float((np.abs(ORD[y] - ORD[pred]) <= 1).mean()),
            "top2": float((top2 == y[:, None]).any(1).mean()),
            "mean_signed_grade_error": float((ORD[pred] - ORD[y]).mean())}


def block(y, p, boot=True):
    pred = p.argmax(1)
    present = sorted(set(y))
    out = {"n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
           "f1_macro": float(f1_score(y, pred, labels=present, average="macro")),
           "bal_acc": float(recall_score(y, pred, labels=present, average="macro"))}
    if len(present) == len(CLASSES):
        out["auc_roc"] = float(roc_auc_score(y, p, multi_class="ovr", average="macro"))
    if boot:
        out["f1_macro_ci"] = ag.bootstrap_ci(y, p, lambda yy, pp: f1_score(yy, pp.argmax(1), labels=present, average="macro")).tolist()
        out["accuracy_ci"] = ag.bootstrap_ci(y, p, lambda yy, pp: accuracy_score(yy, pp.argmax(1))).tolist()
    out |= ordinal(y, p)
    return out


def main():
    # Subconjunto opcional (CSV com colunas arquivo, clean); se ausente, só o teste completo é avaliado.
    cf = RES / "prof_test_originals.csv"
    clean_set = set(pd.read_csv(cf).query("clean").arquivo) if cf.exists() else set()
    results = {}
    for size in ["small", "base", "large"]:
        arq, y, p = load_test(size)
        r = {"test": block(y, p)}
        m = arq.isin(clean_set).to_numpy()
        if m.any():
            r["clean"] = block(y[m], p[m])
        pc = ag.per_class(y, p)
        ci = ag.bootstrap_ci(y, p, lambda yy, pp: [ag.per_class(yy, pp)[c][k] for c in CLASSES for k in ("sens", "spec")])
        for j, (c, k) in enumerate([(c, k) for c in CLASSES for k in ("sens", "spec")]):
            pc[c][f"{k}_ci"] = [float(ci[0][j]), float(ci[1][j])]
        r["per_class"] = pc
        r["confusion_pct"] = (confusion_matrix(y, p.argmax(1), normalize="true") * 100).round(1).tolist()
        cal = {"ece": ag.ece(y, p), "brier": ag.brier(y, p), "nll": ag.nll(y, p),
               "mean_conf_correct": float(p.max(1)[p.argmax(1) == y].mean()),
               "mean_conf_errors": float(p.max(1)[p.argmax(1) != y].mean())}
        val = load_val(size)
        if val is not None:
            T = ag.fit_T(*val)
            pt = ag.temperature(p, T)
            cal |= {"T": float(T), "ece_ts": ag.ece(y, pt), "brier_ts": ag.brier(y, pt), "nll_ts": ag.nll(y, pt),
                    "val_accuracy": float(accuracy_score(val[0], val[1].argmax(1)))}
        r["calibration"] = cal
        # Diagrama de confiabilidade (15 bins) para a figura.
        conf, pred = p.max(1), p.argmax(1)
        edges = np.linspace(0, 1, 16)
        r["reliability"] = [{"lo": float(lo), "hi": float(hi), "n": int(mk.sum()),
                             "acc": float((pred[mk] == y[mk]).mean()) if mk.any() else None,
                             "conf": float(conf[mk].mean()) if mk.any() else None}
                            for lo, hi in zip(edges[:-1], edges[1:]) for mk in [(conf > lo) & (conf <= hi)]]
        results[size] = r

    (RES / "resultados_prof_a5.json").write_text(json.dumps(results, indent=2, default=float))
    L = ["| Modelo | Conjunto | n | Acc [IC95] | F1m [IC95] | AUC | QWK | ±1 grau | top-2 |", "|---|---|---|---|---|---|---|---|---|"]
    for s, r in results.items():
        for k in [k for k in ("test", "clean") if k in r]:
            b = r[k]
            L.append(f"| {s} | {k} | {b['n']} | {b['accuracy']:.3f} [{b['accuracy_ci'][0]:.3f}, {b['accuracy_ci'][1]:.3f}] | "
                     f"{b['f1_macro']:.3f} [{b['f1_macro_ci'][0]:.3f}, {b['f1_macro_ci'][1]:.3f}] | {b.get('auc_roc', float('nan')):.3f} | "
                     f"{b['qwk']:.3f} | {b['within1']:.3f} | {b['top2']:.3f} |")
    L += ["", "| Modelo | ECE | Brier | NLL | T | ECE→TS | Brier→TS | NLL→TS | conf. acertos | conf. erros |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s, r in results.items():
        c = r["calibration"]
        L.append(f"| {s} | {c['ece']:.3f} | {c['brier']:.3f} | {c['nll']:.3f} | {c.get('T', float('nan')):.2f} | "
                 f"{c.get('ece_ts', float('nan')):.3f} | {c.get('brier_ts', float('nan')):.3f} | {c.get('nll_ts', float('nan')):.3f} | "
                 f"{c['mean_conf_correct']:.3f} | {c['mean_conf_errors']:.3f} |")
    for s, r in results.items():
        L += ["", f"### {s}: por classe (teste, n=1050)", "| Classe | Sens. [IC95] | Espec. [IC95] | Precisão | F1 |", "|---|---|---|---|---|"]
        for c, v in r["per_class"].items():
            L.append(f"| {c} | {v['sens']:.3f} [{v['sens_ci'][0]:.2f}, {v['sens_ci'][1]:.2f}] | "
                     f"{v['spec']:.3f} [{v['spec_ci'][0]:.2f}, {v['spec_ci'][1]:.2f}] | {v['prec']:.3f} | {v['f1']:.3f} |")
        L += ["", "Matriz de confusão (% por linha; linhas = real, colunas = predito, ordem " + ", ".join(CLASSES) + "):"]
        L += ["    " + " ".join(f"{x:5.1f}" for x in row) for row in r["confusion_pct"]]
    (RES / "resultados_prof_a5.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
