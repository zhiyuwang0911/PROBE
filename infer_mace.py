#!/usr/bin/env python3
"""MACE PROBE structure classification inference with checkpoint-embedded PCA."""
import argparse
import csv
from pathlib import Path
import torch
from probe.model import PROBEModel
from probe.pca import restore_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mace-model', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--test-xyz', required=True)
    parser.add_argument('--output-dir', default='probe_classifier_inference')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    from probe.backends.mace import (load_mace, get_z_table, load_xyz_dataloader,
                                     process_batch_mace)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'model_config' not in ckpt:
        raise ValueError('Legacy classifier checkpoint has no model_config. Use original inference interface or migrate it.')
    model = restore_model(PROBEModel, ckpt, args.device)
    model.eval()
    extractor = load_mace(args.mace_model, args.device)
    if ckpt.get('original_dim') is not None and ckpt['original_dim'] != extractor.feat_dim:
        raise ValueError('MACE feature dimension differs from checkpoint')
    loader = load_xyz_dataloader(args.test_xyz, get_z_table(extractor),
                float(extractor.mace_model.r_max), batch_size=args.batch_size, shuffle=False)
    rows = []
    for batch in loader:
        feats, mask, pred_e, true_e, n_atoms = process_batch_mace(batch, args.device, extractor)
        with torch.no_grad():
            probs = torch.softmax(model(feats.float(), mask, energy=pred_e.float()), dim=-1)
        for b in range(len(probs)):
            rows.append(dict(structure_index=len(rows), n_atoms=int(n_atoms[b]),
                p_reliable=float(probs[b, 0]), p_unreliable=float(probs[b, 1])))
    if not rows:
        raise ValueError('No structures to infer')
    out = Path(args.output_dir)
    out.mkdir(exist_ok=True, parents=True)
    with (out/'predictions_classifier.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} structures to {out}')

if __name__ == '__main__':
    main()
