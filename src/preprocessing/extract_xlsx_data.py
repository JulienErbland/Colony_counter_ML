import pandas as pd
import glob
import os
import sys
import re

# Find the directory containing this script (e.g. /Colony_counter/src/preprocessing)
# then go two levels up to reach /Colony_counter
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT)

DATA_FOLDER = os.path.join(ROOT, "data", "raw")
SEARCH_PATTERN = os.path.join(DATA_FOLDER, "**", "*.xlsx")
OUTPUT_FILE = os.path.join(ROOT, "data", "global_counting_data.xlsx")

rows = []

# Search all .xlsx files in all subfolders
for file in glob.glob(SEARCH_PATTERN, recursive=True):

    df = pd.read_excel(file, header=None)

    # filename = IBV_File1 or Panama_Plate2 etc.
    folder = os.path.basename(os.path.dirname(file))
    base = os.path.basename(file).replace(".xlsx", "")
    filename = f"{folder}"

    plate_index = 1        # fallback counter
    start_row = 2

    while start_row < df.shape[0]:

        plate_block = df.iloc[start_row:start_row + 3]

        if plate_block.isna().all().all():
            break

        # -------- NEW: read the real plate number from the header line "Plate XX" --------
        # For a block starting at start_row=2, the line "Plate 19" is at index 0 (= start_row-2)
        header_row_idx = start_row - 2
        plate_num = plate_index  # fallback: previous behavior

        if 0 <= header_row_idx < df.shape[0]:
            header_val = str(df.iloc[header_row_idx, 0])
            m = re.search(r"(\d+)", header_val)  # extract the number from "Plate 19"
            if m:
                plate_num = int(m.group(1))
        # ---------------------------------------------------------------------------

        for i in range(3):
            label = str(plate_block.iloc[i, 4])

            if label.lower() == "nan":
                continue

            # logical order 4,3,2,1 → indices 3,2,1,0
            for logical_idx, col_idx in enumerate([3, 2, 1, 0], start=1):

                raw_value = plate_block.iloc[i, col_idx]

                if pd.isna(raw_value):
                    continue

                if str(raw_value).lower() == "x":
                    value = -1
                else:
                    value = raw_value

                name = f"{filename}_plate{plate_num}_{label}{logical_idx}"

                rows.append({
                    "Name": name,
                    "Value": value
                })

        plate_index += 1
        start_row += 6

result_df = pd.DataFrame(rows)
result_df.to_excel(OUTPUT_FILE, index=False)
