import json
import numpy as np

with open('dataset/nextgqa/choice_grounder_outputs_test.json') as f:
    data = json.load(f)

def iou(ps, pe, gs, ge):
    inter = max(0, min(pe, ge) - max(ps, gs))
    union = max(pe, ge) - min(ps, gs)
    return inter / union if union > 0 else 0.0

def iop(ps, pe, gs, ge):
    inter = max(0, min(pe, ge) - max(ps, gs))
    pred_len = pe - ps
    return inter / pred_len if pred_len > 0 else 0.0

def best_iou_iop(ps, pe, gt_segs):
    bi, bp = 0.0, 0.0
    for gs, ge in gt_segs:
        bi = max(bi, iou(ps, pe, gs, ge))
        bp = max(bp, iop(ps, pe, gs, ge))
    return bi, bp

thresholds = [0.3, 0.5, 0.7]

buckets = {
    'oracle': dict(ious=[], iops=[], riu=np.zeros(3), rip=np.zeros(3)),
    'pred':   dict(ious=[], iops=[], riu=np.zeros(3), rip=np.zeros(3)),
    'best5':  dict(ious=[], iops=[], riu=np.zeros(3), rip=np.zeros(3)),
}

n_skip = 0
for r in data:
    gt = r.get('gt_segments')
    props = r.get('top5_proposals', [])
    if not gt or not props or len(props) < 5:
        n_skip += 1
        continue

    ans_idx  = r['answer_idx']
    pred_idx = r['blip2_pred_idx']
    duration = r.get('duration', 1.0)

    cases = {
        'oracle': props[ans_idx],
        'pred':   props[pred_idx],
    }

    for key, p in cases.items():
        bi, bp = best_iou_iop(p[0], p[1], gt)
        buckets[key]['ious'].append(bi)
        buckets[key]['iops'].append(bp)
        for k, th in enumerate(thresholds):
            if bi >= th: buckets[key]['riu'][k] += 1
            if bp >= th: buckets[key]['rip'][k] += 1

    # Best of 5
    best_bi = max(best_iou_iop(props[i][0], props[i][1], gt)[0] for i in range(len(props)))
    best_bp = max(best_iou_iop(props[i][0], props[i][1], gt)[1] for i in range(len(props)))
    buckets['best5']['ious'].append(best_bi)
    buckets['best5']['iops'].append(best_bp)
    for k, th in enumerate(thresholds):
        if best_bi >= th: buckets['best5']['riu'][k] += 1
        if best_bp >= th: buckets['best5']['rip'][k] += 1

ng = len(buckets['oracle']['ious'])
print(f"Evaluated: {ng} / {len(data)} (skip {n_skip} no GT)")
print()

header = f"{'':24} {'mIoU':>7} {'mIoP':>7}  {'@.3':>6} {'@.5':>6} {'@.7':>6}  {'iop@.3':>7} {'iop@.5':>7} {'iop@.7':>7}"
print(header)
print('-' * 90)

labels = {
    'oracle': 'Oracle (GT choice)',
    'pred':   'BLIP-2 pred choice',
    'best5':  'Best-of-5 (upper bound)',
}
for key in ['oracle', 'pred', 'best5']:
    b = buckets[key]
    n = ng
    miu = np.mean(b['ious']) * 100
    mip = np.mean(b['iops']) * 100
    riu = b['riu'] / n * 100
    rip = b['rip'] / n * 100
    print(f"  {labels[key]:<22} {miu:>7.2f}% {mip:>7.2f}%  {riu[0]:>6.2f}% {riu[1]:>6.2f}% {riu[2]:>6.2f}%  "
          f"{rip[0]:>7.2f}% {rip[1]:>7.2f}% {rip[2]:>7.2f}%")
