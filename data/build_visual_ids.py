import os
import json
import hashlib
from collections import defaultdict

import numpy as np
from datasets import load_dataset, concatenate_datasets


DATASET_NAME = "luigi-s/EEG_Image_CVPR_ALL_subj"
OUTPUT_DIR = "visual_ids"
OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    DATASET_NAME.replace("/", "_"),
    "visual_ids_by_subject.json",
)


def hash_image_from_example(example):
    img = example["image"]
    arr = np.asarray(img)

    # Include shape and dtype to avoid rare collisions from raw bytes alone
    h = hashlib.sha1()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    ds_train = load_dataset(DATASET_NAME, split="train")
    ds_val = load_dataset(DATASET_NAME, split="validation")
    ds_test = load_dataset(DATASET_NAME, split="test")

    ds_train = ds_train.add_column("__hf_split__", ["train"] * len(ds_train))
    ds_val = ds_val.add_column("__hf_split__", ["validation"] * len(ds_val))
    ds_test = ds_test.add_column("__hf_split__", ["test"] * len(ds_test))

    full_data = concatenate_datasets([ds_train, ds_val, ds_test])

    subject_to_indices = defaultdict(list)
    for global_idx, subj in enumerate(full_data["subject"]):
        subject_to_indices[int(subj)].append(global_idx)

    hash_to_visual_id = {}
    visual_ids_by_subject = {}
    stats_by_subject = {}

    next_visual_id = 0

    for subj in sorted(subject_to_indices.keys()):
        visual_ids = []
        hashes = []

        print(f"[SUBJ {subj}] processing {len(subject_to_indices[subj])} samples")

        for local_idx, global_idx in enumerate(subject_to_indices[subj]):
            ex = full_data[global_idx]
            img_hash = hash_image_from_example(ex)

            if img_hash not in hash_to_visual_id:
                hash_to_visual_id[img_hash] = next_visual_id
                next_visual_id += 1

            visual_id = hash_to_visual_id[img_hash]
            visual_ids.append(int(visual_id))
            hashes.append(img_hash)

        visual_ids_by_subject[str(subj)] = visual_ids
        stats_by_subject[str(subj)] = {
            "num_samples": len(visual_ids),
            "num_unique_visual_ids": len(set(visual_ids)),
        }

    payload = {
        "dataset_name": DATASET_NAME,
        "num_unique_images_global": int(next_visual_id),
        "num_total_samples": int(len(full_data)),
        "visual_ids_by_subject": visual_ids_by_subject,
        "stats_by_subject": stats_by_subject,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n[DONE]")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Global unique image ids: {next_visual_id}")

    # Extra sanity check: count how many visual IDs appear in multiple subjects
    visual_id_to_subjects = defaultdict(set)
    for subj, ids in visual_ids_by_subject.items():
        for vid in ids:
            visual_id_to_subjects[int(vid)].add(int(subj))

    multi_subject = sum(1 for s in visual_id_to_subjects.values() if len(s) > 1)
    print(f"Visual IDs appearing in >=2 subjects: {multi_subject}")


if __name__ == "__main__":
    main()