"""Data for the diagnostic figures, collected from the trained model."""

from typing import Dict

import numpy as np
import torch

from .decision import PerSampleDecisionModule
from .encoding import poisson_encode


@torch.no_grad()
def collect_diagnostics(model, tasks, device, cfg, si=None,
                        isi_est=None, max_batches: int = 8) -> Dict:
    """One evaluation pass over the trained model for the diagnostic figures.

    Stores, per task, the entropy of correctly and incorrectly classified test
    samples, attention and firing rates, the spike rasters of one sample, and
    for the whole run the threshold and membrane traces and the SI importance
    maps. At most `max_batches` test batches per task are used.
    """
    model.eval()
    D: Dict = {"per_task": [], "samples": []}
    n_sub = 1500                       # cap per-task arrays kept in the JSON

    for t in tasks:
        tid = t["task_id"]
        H_ok, H_bad, ATT, R1, R2 = [], [], [], [], []
        first = True
        for bi, (x, y) in enumerate(t["test_loader"]):
            if bi >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            sp = poisson_encode(x, cfg.t_steps)
            logits, info = model(sp, task_id=tid if cfg.multi_head else 0,
                                 collect_raster=first)
            H = PerSampleDecisionModule.entropy(logits)
            att = PerSampleDecisionModule.attention(H, info["rate_l1"], info["rate_l2"])
            ok = (logits.argmax(1) == y)
            H_ok += H[ok].cpu().tolist()
            H_bad += H[~ok].cpu().tolist()
            ATT += att.cpu().tolist()
            R1 += info["rate_l1"].cpu().tolist()
            R2 += info["rate_l2"].cpu().tolist()

            if first:
                first = False
                D["samples"].append({
                    "task_id": tid,
                    "digits": list(t["digits"]),
                    "image": x[0, 0].cpu().tolist(),
                    "input_raster": sp[:, 0, :].cpu().numpy()[:, :120].tolist(),
                    "raster_l1": info["spikes_l1"][:, 0, :].cpu().numpy()[:, :96].tolist(),
                    "raster_l2": info["spikes_l2"][:, 0, :].cpu().numpy()[:, :96].tolist(),
                })
                if isi_est is not None and tid == len(tasks) - 1:
                    cv1, _ = isi_est.neuron_importance(info["spikes_l1"])
                    D["isi_importance_l1"] = cv1.tolist()
                    cv2, _ = isi_est.neuron_importance(info["spikes_l2"])
                    D["isi_importance_l2"] = cv2.tolist()

        D["per_task"].append({
            "task_id": tid,
            "entropy_correct": H_ok[:n_sub],
            "entropy_wrong": H_bad[:n_sub],
            "attention": ATT[:n_sub],
            "rate_l1": R1[:n_sub],
            "rate_l2": R2[:n_sub],
            "n_correct": len(H_ok), "n_wrong": len(H_bad),
            "acc": len(H_ok) / max(len(H_ok) + len(H_bad), 1),
            "entropy_correct_mean": float(np.mean(H_ok)) if H_ok else 0.0,
            "entropy_wrong_mean": float(np.mean(H_bad)) if H_bad else 0.0,
        })

    # ---- ALIF threshold trajectory on a real batch ------------------------
    x, _ = next(iter(tasks[-1]["test_loader"]))
    sp = poisson_encode(x[:32].to(device), cfg.t_steps)
    Tn = cfg.t_steps
    s1 = model.cell1.init_state(sp.shape[1], device)
    s2 = model.cell2.init_state(sp.shape[1], device)
    vth_tr, v_tr, rate_tr = [], [], []
    for tt in range(Tn):
        z1, s1, vth1 = model.cell1(model.fc1(sp[tt]), s1)
        z2, s2, _ = model.cell2(model.fc2(z1), s2)
        vth_tr.append(float(vth1.mean()))
        v_tr.append(float(s1.v.mean()))
        rate_tr.append(float(z1.mean()))
    D["threshold_trace"] = vth_tr
    D["membrane_trace"] = v_tr
    D["rate_trace"] = rate_tr
    D["membrane_hist_l1"] = s1.v.flatten().cpu().numpy()[:4000].tolist()

    # ---- SI importance maps ----------------------------------------------
    if si is not None:
        om = si.neuron_omega()
        for k in ("l1", "l2"):
            if k in om:
                D[f"omega_neuron_{k}"] = om[k].tolist()
        nm = si.normalised_omega_map()
        if "fc1.weight" in nm:
            M = nm["fc1.weight"].cpu().numpy()
            step = max(1, M.shape[1] // 96)
            D["omega_map_l1"] = M[:96, ::step][:, :96].tolist()
        if "fc2.weight" in nm:
            M = nm["fc2.weight"].cpu().numpy()
            D["omega_map_l2"] = M[:96, :96].tolist()

    model.train()
    return D
