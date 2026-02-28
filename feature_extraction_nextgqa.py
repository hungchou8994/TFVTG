"""
NExT-GQA BLIP-2 Feature Extraction
====================================
Extracts BLIP-2 visual features for NExT-GQA videos (sourced from VidOR dataset).

VidOR folder structure:
    {vidor_root}/{folder}/{video_id}/{video_id}.mp4
    e.g.  /path/to/vidor/1020/13496784364/13496784364.mp4

Output:
    {save_root}/{video_id}.npy   shape [T, 32, 256]  (float16)

Usage:
    python feature_extraction_nextgqa.py \\
        --vidor_root /path/to/vidor_videos \\
        --save_root  dataset/nextgqa/blip2_features \\
        --splits val test \\
        --fps 3

Colab example:
    !python feature_extraction_nextgqa.py \\
        --vidor_root /content/drive/MyDrive/vidor_videos \\
        --save_root  dataset/nextgqa/blip2_features \\
        --splits val test
"""

import os
import csv
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from tqdm import tqdm


# ── BLIP-2 model (loaded once at startup) ────────────────────────────────────

DEVICE = 'cuda'

def load_blip2():
    from lavis.models import load_model_and_preprocess
    model, vis_processors, _ = load_model_and_preprocess(
        "blip2_image_text_matching", "coco", device=DEVICE, is_eval=True
    )
    vis_proc = transforms.Compose([
        t for t in vis_processors['eval'].transform.transforms
        if not isinstance(t, transforms.ToTensor)
    ])
    return model, vis_proc


# ── Video loading ─────────────────────────────────────────────────────────────

def load_video_frames(video_path, fps=3):
    """Load video and sample frames at given fps. Returns float tensor [T,3,H,W]."""
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, num_threads=1, ctx=cpu(0))
    duration = len(vr) / vr.get_avg_fps()
    num_frames = max(1, round(duration * fps))
    indices = np.linspace(0, len(vr) - 1, num=num_frames).round().astype(np.int32)
    vr.seek(0)
    frames = vr.get_batch(indices).permute(0, 3, 1, 2) / 255.0  # [T,3,H,W]
    return frames


# ── Feature extraction ────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(model, vis_proc, video_path, fps=3, batch_size=128):
    """Extract BLIP-2 features for a single video. Returns numpy [T, 32, 256] float16."""
    frames = load_video_frames(video_path, fps=fps)   # [T,3,H,W]
    imgs = vis_proc(frames)                            # [T,3,224,224]

    all_feats = []
    for start in range(0, imgs.size(0), batch_size):
        batch = imgs[start:start + batch_size].to(DEVICE)
        with model.maybe_autocast():
            image_embeds = model.ln_vision(model.visual_encoder(batch))
        image_embeds = image_embeds.float()
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=DEVICE)
        query_tokens = model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = model.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        feats = model.vision_proj(query_output.last_hidden_state)  # [B, 32, 256]
        all_feats.append(feats.cpu().half())

    return torch.cat(all_feats, dim=0).numpy()  # [T, 32, 256]


# ── Path resolution ───────────────────────────────────────────────────────────

VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mkv', '.mov', '.webm']

def find_video_path(vidor_root, vidor_rel):
    """
    Try to find video file given VidOR relative path.
    vidor_rel example: '1020/13496784364'

    Tries in order:
      1. {vidor_root}/{folder}/{video_id}.ext      ← flat (most common)
      2. {vidor_root}/{folder}/{video_id}/{video_id}.ext  ← VidOR original nested
      3. {vidor_root}/{video_id}.ext               ← fully flat fallback
    """
    folder, video_id = vidor_rel.split('/')

    # 1. Flat inside folder: {folder}/{video_id}.ext  (most common setup)
    for ext in VIDEO_EXTENSIONS:
        candidate = os.path.join(vidor_root, folder, video_id + ext)
        if os.path.isfile(candidate):
            return candidate

    # 2. VidOR original nested: {folder}/{video_id}/{video_id}.ext
    video_dir = os.path.join(vidor_root, folder, video_id)
    if os.path.isdir(video_dir):
        for ext in VIDEO_EXTENSIONS:
            candidate = os.path.join(video_dir, video_id + ext)
            if os.path.isfile(candidate):
                return candidate
        # Any video file in subdirectory
        for fname in os.listdir(video_dir):
            if any(fname.endswith(ext) for ext in VIDEO_EXTENSIONS):
                return os.path.join(video_dir, fname)

    # 3. Fully flat: {vidor_root}/{video_id}.ext
    for ext in VIDEO_EXTENSIONS:
        candidate = os.path.join(vidor_root, video_id + ext)
        if os.path.isfile(candidate):
            return candidate

    return None


# ── Collect video IDs for requested splits ────────────────────────────────────

def collect_video_ids(splits):
    """Return sorted list of unique video_ids across the requested splits."""
    video_ids = set()
    for split in splits:
        csv_path = f'dataset/nextgqa/{split}.csv'
        if not os.path.exists(csv_path):
            print(f'[WARN] CSV not found: {csv_path} — skipping split "{split}"')
            continue
        with open(csv_path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                video_ids.add(row['video_id'])
    return sorted(video_ids)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Extract BLIP-2 features for NExT-GQA (VidOR videos)')
    parser.add_argument('--vidor_root', required=True,
                        help='Root folder of VidOR videos (contains subfolders like 0101/, 1020/, ...)')
    parser.add_argument('--save_root', default='dataset/nextgqa/blip2_features',
                        help='Output folder for .npy features')
    parser.add_argument('--map_file', default='dataset/nextgqa/map_vid_vidorID.json',
                        help='Mapping: video_id -> VidOR relative path')
    parser.add_argument('--splits', nargs='+', default=['val', 'test'],
                        choices=['val', 'test'],
                        help='Which splits to process')
    parser.add_argument('--fps', type=float, default=3.0,
                        help='Frames per second to sample (default: 3)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='BLIP-2 batch size per GPU forward pass')
    args = parser.parse_args()

    os.makedirs(args.save_root, exist_ok=True)

    # Load VidOR mapping
    with open(args.map_file) as f:
        vid_to_vidor = json.load(f)
    print(f'VidOR mapping loaded: {len(vid_to_vidor)} entries')

    # Collect video IDs
    video_ids = collect_video_ids(args.splits)
    print(f'Splits {args.splits}: {len(video_ids)} unique videos')

    # Separate already-done and pending
    pending, already_done, missing_map, missing_file = [], [], [], []
    for vid in video_ids:
        save_path = os.path.join(args.save_root, vid + '.npy')
        if os.path.exists(save_path):
            already_done.append(vid)
            continue
        if vid not in vid_to_vidor:
            missing_map.append(vid)
            continue
        video_path = find_video_path(args.vidor_root, vid_to_vidor[vid])
        if video_path is None:
            missing_file.append(vid)
            continue
        pending.append((vid, video_path, save_path))

    print(f'  Already extracted : {len(already_done)}')
    print(f'  To extract        : {len(pending)}')
    if missing_map:
        print(f'  [WARN] No VidOR mapping for {len(missing_map)} videos: {missing_map[:5]}...')
    if missing_file:
        print(f'  [WARN] Video file not found for {len(missing_file)} videos: {missing_file[:5]}...')

    if not pending:
        print('Nothing to do. All features already extracted.')
        return

    # Load BLIP-2
    print('\nLoading BLIP-2 model...')
    model, vis_proc = load_blip2()
    print('BLIP-2 loaded.\n')

    # Extract
    failed = []
    pbar = tqdm(pending, desc='Extracting features')
    for vid, video_path, save_path in pbar:
        pbar.set_postfix({'vid': vid[:14]})

        # Auto-retry with smaller batch_size on CUDA OOM
        batch_size = args.batch_size
        success = False
        last_error = None
        while batch_size >= 8:
            try:
                torch.cuda.empty_cache()
                feats = extract_features(model, vis_proc, video_path,
                                         fps=args.fps, batch_size=batch_size)
                np.save(save_path, feats)
                success = True
                break
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    tqdm.write(f'  [OOM] {vid}: batch_size={batch_size} → retry with {batch_size // 2}')
                    torch.cuda.empty_cache()
                    batch_size //= 2
                    last_error = e
                else:
                    last_error = e
                    break
            except Exception as e:
                last_error = e
                break

        if not success:
            tqdm.write(f'  [ERROR] {vid}: {last_error}')
            failed.append({'video_id': vid, 'path': video_path, 'error': str(last_error)})

    print(f'\n{"="*50}')
    print(f'Done!')
    print(f'  Extracted  : {len(pending) - len(failed)}')
    print(f'  Failed     : {len(failed)}')
    if failed:
        fail_path = os.path.join(args.save_root, 'failed_videos.json')
        with open(fail_path, 'w') as f:
            json.dump(failed, f, indent=2)
        print(f'  Failed list: {fail_path}')
    print(f'{"="*50}')


if __name__ == '__main__':
    main()
