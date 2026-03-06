import json
import numpy as np

with open('dataset/nextgqa/choice_verifier_outputs_test.json') as f:
    data = json.load(f)

done = [r for r in data if 'best_proposal' in r]
print(f'Evaluated: {len(done)} / {len(data)}')

def iou(ps, pe, gs, ge):
    inter = max(0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0

def iop(ps, pe, gs, ge):
    inter = max(0, min(pe, ge) - max(ps, gs))
    pred_len = pe - ps
    return inter / pred_len if pred_len > 0 else 0.0

def best_scores(ps, pe, gt_segs):
    bi, bp = 0.0, 0.0
    for gs, ge in gt_segs:
        bi = max(bi, iou(ps, pe, gs, ge))
        bp = max(bp, iop(ps, pe, gs, ge))
    return bi, bp

thresholds = [0.3, 0.5, 0.7]

# Per-type buckets
types = ['CH', 'CW', 'TC', 'TN', 'TP']
stats = {t: dict(n=0, ious=[], iops=[], riu=np.zeros(3), rip=np.zeros(3)) for t in types + ['ALL']}

for r in done:
    gt = r.get('gt_segments', [])
    bp = r['best_proposal']
    bi, bip = best_scores(bp[0], bp[1], gt) if gt else (0.0, 0.0)

    qtype = r.get('type', 'ALL')
    for key in [qtype, 'ALL']:
        s = stats[key]
        s['n']   += 1
        s['ious'].append(bi)
        s['iops'].append(bip)
        for k, th in enumerate(thresholds):
            if bi  >= th: s['riu'][k] += 1
            if bip >= th: s['rip'][k] += 1

print()
print(f"  {'Type':<4} {'N':>5}  {'mIoU':>6}  {'mIoP':>6}  {'IoU@.3':>7} {'IoU@.5':>7} {'IoU@.7':>7}  {'IoP@.3':>7} {'IoP@.5':>7} {'IoP@.7':>7}")
print('-' * 90)

for key in types + ['ALL']:
    s = stats[key]
    n = s['n']
    if n == 0:
        continue
    miu  = np.mean(s['ious']) * 100
    mip  = np.mean(s['iops']) * 100
    riu  = s['riu'] / n * 100
    rip  = s['rip'] / n * 100
    print(f"  {key:<4} {n:>5}  {miu:>6.2f}%  {mip:>6.2f}%  "
          f"{riu[0]:>7.2f}% {riu[1]:>7.2f}% {riu[2]:>7.2f}%  "
          f"{rip[0]:>7.2f}% {rip[1]:>7.2f}% {rip[2]:>7.2f}%")
