from __future__ import annotations

from pathlib import Path
import os
import sys
import torch
import pandas as pd
import torch.nn.functional as F
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.preprocessing import crop_plates
from src.preprocessing.crop_wells import crop_wells_script 
from src.ml.models.CountabilityClassifier import CountabilityClassifier
from src.ml.data.dataset import ColonyDataset
from src.ml.data.transforms import get_classifier_test_transforms,get_counter_test_transforms
from src.ml.models.ResNet34 import ResNet34Regressor
from src.utils.write_xlsx import write_results_xlsx

SRC_ROOT = Path(os.path.join(ROOT, "usage", "images"))
DST_PLATES_ROOT = Path(os.path.join(ROOT, "usage", "processed_images", "cropped_plates"))
DST_WELLS_ROOT = Path(os.path.join(ROOT, "usage", "processed_images", "cropped_wells")) 
PROCESSED_IMAGES_ROOT = Path(os.path.join(ROOT, "usage", "processed_images"))
RESULTS_PATH = Path(os.path.join(ROOT, "usage", "results.xlsx"))
COUNTABILITY_WEIGHTS_PATH = Path(os.path.join(ROOT, "countability_classifier.pth"))
COUNTER_WEIGHTS_PATH = Path(os.path.join(ROOT, "resnet34_colony.pth"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    # 1) Crop Plates
    if not crop_plates.process_folder(SRC_ROOT, DST_PLATES_ROOT):
        print(
            "Please check the logs and crop the problematic images manually.\n"
            "THE IMAGE SHOULD ONLY CONTAIN THE PLATE WITH THE 12 WELLS AND HAVE GOOD QUALITY."
        )
        # Cleanup: remove usage/processed_images folder
        if PROCESSED_IMAGES_ROOT.exists():
            shutil.rmtree(PROCESSED_IMAGES_ROOT)
        return

    # 2) Crop Wells
    if not crop_wells_script(DST_PLATES_ROOT, DST_WELLS_ROOT):
        print(
            " - 1) verify that the image names start with \"plate_x\" (x being a number)\n"
            " - 2) verify that the images are 12-well plates and are cropped correctly\n"
            " - 3) ensure that the image quality is not poor or blurry\n"
            " - 4) check the logs for detailed errors"
        )
        # Cleanup: remove usage/processed_images folder
        if PROCESSED_IMAGES_ROOT.exists():
            shutil.rmtree(PROCESSED_IMAGES_ROOT)
        return

    # 3) Build dataframe with one row per well image (dummy labels for inference)
    image_paths = sorted(
        [p for p in DST_WELLS_ROOT.rglob("*.png")]
        + [p for p in DST_WELLS_ROOT.rglob("*.jpg")]
        + [p for p in DST_WELLS_ROOT.rglob("*.jpeg")]
    )
    df = pd.DataFrame(
        {
            "name": [p.name for p in image_paths],
            "path": [str(p) for p in image_paths],          # column name expected by ColonyDataset
            "value": [0] * len(image_paths),                # dummy labels (not used at inference)
            "is_countable": [0] * len(image_paths),         # dummy labels (not used at inference)
        }
    )
    # 4) PyTorch Datasets
    ds_classifier = ColonyDataset(
        df=df,
        transform=get_classifier_test_transforms(),
    )

    ds_counter = ColonyDataset(
        df=df,
        transform=get_counter_test_transforms(),
    )

    # 5) Models
    classifier = CountabilityClassifier(backbone="efficientnet_b0", pretrained=False)
    classifier.load_state_dict(torch.load(COUNTABILITY_WEIGHTS_PATH, map_location=device, weights_only=True))
    classifier.to(device)
    classifier.eval()

    counter = ResNet34Regressor(pretrained=True, dropout_p=0.5, freeze_backbone=False)
    counter.load_state_dict(torch.load(COUNTER_WEIGHTS_PATH, map_location=device, weights_only=True))
    counter.to(device)
    counter.eval()

    
    # 6) Inference loop
    print("\nCounting the plates...\n")
    records = []

    with torch.no_grad():
        for global_idx in range(len(df)):
            
            # --- 1. Countability classification ---
            
            img, _ = ds_classifier[global_idx]
            img = img.to(device).unsqueeze(0)  # [1, C, H, W] --> tensor shape

            logits = classifier(img)                 
            probs = F.softmax(logits, dim=1)
            countable_score = probs[0, 1].item()
            pred_is_countable = int(countable_score >= 0.5) # threshold = 0.5

            df_row = df.iloc[global_idx]
            record = {
                "name": df_row["name"],
                "pred_is_countable": pred_is_countable,
            }

            # --- 2. Counter ---

            if pred_is_countable == 1:

                img, _ = ds_counter[global_idx]         
                img = img.unsqueeze(0).to(device)

                raw_pred = counter(img).item()
                predicted_count = int(max(0, round(raw_pred)))

                record["predicted_count"] = predicted_count
            else:
                record["predicted_count"] = -1      # -1 = uncountable

            records.append(record)

    # 7) Save results to Excel in results.xlsx
    write_results_xlsx(records, RESULTS_PATH)

    print("Results saved to:", RESULTS_PATH)

    # 8) Cleanup: remove usage/processed_images folder
    if PROCESSED_IMAGES_ROOT.exists():
        shutil.rmtree(PROCESSED_IMAGES_ROOT)

    print("SUCCESS ! You can now check the results.xlsx file.")


if __name__ == "__main__":
    main()
