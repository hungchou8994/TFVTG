import json

with open('dataset/nextgqa/choice_verifier_outputs_test.json') as f:
    data = json.load(f)

done = [r for r in data if 'best_proposal' in r]
print(f'Done: {len(done)}/{len(data)}')
correct = 0
for r in done:
    rank = r.get('gemini_ranking', [])
    pred_choice = rank[0] - 1  # 0-based
    gt = r['answer_idx']
    match = pred_choice == gt
    if match:
        correct += 1
    choices = r.get('choices', [])
    print(f'  qid={r["qid"]} | ranking={rank} -> pred={pred_choice} | gt={gt} | {"CORRECT" if match else "WRONG"}')
    print(f'    Q: {r["question"]}')
    if choices:
        for i, c in enumerate(choices):
            marker = '<-- GT' if i == gt else ('  <- pred' if i == pred_choice else '')
            print(f'    [{i}] {c}  {marker}')
    print(f'    Reasoning: {r.get("verifier_reasoning","")}')
    print()

if done:
    print(f'Acc (dry run): {correct}/{len(done)} = {correct/len(done)*100:.1f}%')
