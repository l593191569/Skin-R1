import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
import tqdm
import yaml

from MONET.utils.io import get_hdf5_key, load_pkl

# =====================
# Text preprocessing
# =====================
def process_text(text_raw):
    """
    Normalize raw text: remove newlines/tabs/special symbols, lowercase,
    strip figure/fig/efig prefixes, and strip leading numbers and symbols.
    """
    text_processed = (
        text_raw.strip().replace("\n", " ").replace("\t", " ").replace("•", " ").replace("", " ")
    )
    text_processed = text_processed.replace("  ", " ").replace("  ", " ").replace("  ", " ")
    text_processed = text_processed.lower().strip()
    # print(text_processed)

    if text_processed.startswith("figure"):
        text_processed = re.sub(r"\Afigure", "", text_processed)
    elif text_processed.startswith("efig"):
        text_processed = re.sub(r"\Aefig", "", text_processed)
    elif text_processed.startswith("fig"):
        text_processed = re.sub(r"\Afig", "", text_processed)

    text_processed = text_processed.strip()
    text_processed = re.sub(r"\A[\-\.\s0-9\:]+", "", text_processed)
    # text_processed = text_processed.split("(courtesy")[0]
    text_processed = text_processed.strip()

    return text_processed

# =====================
# Legend-text cleanup
# =====================
def text_remove_legend(text_raw):
    """
    Remove legend markers (e.g. (black arrow), (red circle)) and keep only the core description.
    """
    for color in ["black ", "yellow ", "red ", "white ", "gay ", ""]:
        for shape in [
            "arrow",
            "arrows",
            "box",
            "boxes",
            "clrcle",
            "circle",
            "circles",
            "star",
            "stars",
            "dahsed line",
            "dahsed lines",
            "dotted line",
            "dotted lines",
        ]:
            for left_paren in ["(", "{"]:
                for right_paren in [")", "}"]:
                    text_raw = text_raw.replace(f"{left_paren}{color}{shape}{right_paren}", "")
    if text_raw.count("(") == 1:
        text_raw = text_raw.split("(")[0]
    return text_raw

def compute_spatial_features(text_info_df, image_info):
    image_bbox = image_info["bbox"]

    def horizontal_edge_dist(text_box, image_box):
        if text_box[2] < image_box[0]:  # left
            return image_box[0] - text_box[2]
        elif text_box[0] > image_box[2]:  # right
            return text_box[0] - image_box[2]
        else:  # overlapping
            return 0

    def vertical_edge_dist(text_box, image_box):
        if text_box[3] < image_box[1]:  # above
            return image_box[1] - text_box[3]
        elif text_box[1] > image_box[3]:  # below
            return text_box[1] - image_box[3]
        else:
            return 0

    text_info_df = text_info_df.copy()
    text_info_df["edge_dist_x"] = text_info_df["bbox"].apply(lambda x: horizontal_edge_dist(x, image_bbox))
    text_info_df["edge_dist_y"] = text_info_df["bbox"].apply(lambda x: vertical_edge_dist(x, image_bbox))
    text_info_df["edge_dist"] = (
        text_info_df["edge_dist_x"] ** 2 + text_info_df["edge_dist_y"] ** 2
    ) ** 0.5
    text_info_df["text_under_image"] = text_info_df["bbox"].apply(
        lambda x: x[1] >= image_bbox[3]
    )
    text_info_df["image_bbox"] = [image_bbox] * len(text_info_df)

    return text_info_df


# def extract_label(text_info_df, label_patterns, label_dist_threshold=30):
#     text_blocks_sorted = text_info_df.sort_values("edge_dist", ascending=True)
#     if len(text_blocks_sorted) > 0 and text_blocks_sorted.iloc[0]["edge_dist"] < label_dist_threshold:
#         for pat in label_patterns:
#             m = re.search(pat, text_blocks_sorted.iloc[0]["text"], re.IGNORECASE)
#             if m:
#                 return m.group(1)
#     return None

def extract_label_and_sublabel(text_info_df, label_patterns, label_dist_threshold=30):
    text_blocks_sorted = text_info_df.sort_values("edge_dist", ascending=True)
    sublabel = None
    mainlabel = None
    mainlabel_bbox = None  # bbox of the text block holding the main label
    mainlabel_text = None  # content of the text block holding the main label
    # print("extract_label_and_sublabel--------------------------------")
    if len(text_blocks_sorted) > 0 and text_blocks_sorted.iloc[0]["edge_dist"] < label_dist_threshold:
        text0 = text_blocks_sorted.iloc[0]["text"].strip()
        print(f"[DEBUG] text0=<{text0}>")
        # Only consider two space-separated uppercase letters
        # m = re.fullmatch(r"([A-Z])\s+([A-Z])", text0)
        # if m:
        #     letters = [m.group(1), m.group(2)]
        #     text_bbox = text_blocks_sorted.iloc[0]["bbox"]
        #     x0, y0, x1, y1 = text_bbox
        #     region_width = (x1 - x0) / 2
        #     # Get the image bbox
        #     image_bbox = text_blocks_sorted.iloc[0]["image_bbox"] if "image_bbox" in text_blocks_sorted.iloc[0] else None
        #     if image_bbox is None and "image_bbox" in text_info_df.columns:
        #         image_bbox = text_info_df.iloc[0]["image_bbox"]
        #     img_center_x = None
        #     if image_bbox is not None:
        #         img_center_x = (image_bbox[0] + image_bbox[2]) / 2
        #     # New logic: assign the sublabel using the nearest region center
        #     region_centers = [x0 + region_width/2, x0 + region_width + region_width/2]
        #     if image_bbox is not None and img_center_x is not None:
        #         dists = [abs(img_center_x - rc) for rc in region_centers]
        #         sublabel = letters[dists.index(min(dists))]
        #     print(f"[DEBUG] letters={letters}, text_bbox={text_bbox}, image_bbox={image_bbox}, img_center_x={img_center_x}")
        #     print(f"[DEBUG] region_centers: {region_centers}")
        #     print(f"[DEBUG] sublabel={sublabel}")
        #     # Find the main label (below the image and horizontally overlapping)
        #     if image_bbox is not None:
        #         for _, row in text_info_df.iterrows():
        #             tbbox = row["bbox"]
        #             if tbbox[1] < image_bbox[3]:
        #                 continue
        #             overlap = min(tbbox[2], image_bbox[2]) - max(tbbox[0], image_bbox[0])
        #             if overlap <= 0:
        #                 continue
        #             overlap_ratio = overlap / (image_bbox[2] - image_bbox[0])
        #             if overlap_ratio < 0.5:
        #                 continue
        #             for pat in label_patterns:
        #                 m = re.search(pat, row["text"], re.IGNORECASE)
        #                 if m:
        #                     mainlabel = m.group(1)
        #                     break
        #             if mainlabel:
        #                 break
        # el
        if re.fullmatch(r"[A-Z]", text0):
            sublabel = text0
            print(f"[DEBUG] sublabel(single)={sublabel}")
            print(f"[DEBUG] text_blocks_sorted shape: {text_blocks_sorted.shape}")
            print(f"[DEBUG] text_blocks_sorted columns: {text_blocks_sorted.columns.tolist()}")
            print(f"[DEBUG] text_blocks_sorted=\n{text_blocks_sorted.to_string()}")
            # Find the main label (below the image and horizontally overlapping)
            image_bbox = text_blocks_sorted.iloc[0]["image_bbox"] if "image_bbox" in text_blocks_sorted.iloc[0] else None
            if image_bbox is None:
                print(f"[DEBUG] image_bbox is None")
            else:
                print(f"[DEBUG] Looking for the main label, image bbox: {image_bbox}")
                for idx, row in text_blocks_sorted.iterrows():
                    tbbox = row["bbox"]
                    print(f"[DEBUG] Checking text block {idx}: bbox={tbbox}, text='{row['text'][:50]}...'")
                    
                    if tbbox[1] < image_bbox[3]:
                        print(f"[DEBUG] Text block {idx} is above the image, skipping")
                        continue
                    
                    overlap = min(tbbox[2], image_bbox[2]) - max(tbbox[0], image_bbox[0])
                    print(f"[DEBUG] Text block {idx} overlap: min({tbbox[2]}, {image_bbox[2]}) - max({tbbox[0]}, {image_bbox[0]}) = {overlap}")
                    
                    if overlap <= 0:
                        print(f"[DEBUG] Text block {idx} has no horizontal overlap, skipping")
                        continue
                    
                    # overlap_ratio = overlap / (image_bbox[2] - image_bbox[0])
                    # if overlap_ratio < 0.5:
                    #     continue
                    
                    print(f"[DEBUG] Text block {idx} passed the spatial check, running regex matching")
                    for pat in label_patterns:
                        m = re.search(pat, row["text"], re.IGNORECASE)
                        if m:
                            mainlabel = m.group(1)
                            mainlabel_bbox = row["bbox"]  # bbox of the text block holding the main label
                            mainlabel_text = row["text"]  # content of the text block holding the main label
                            print(f"[DEBUG] Text block {idx} matched the main label: {mainlabel}")
                            break
                    if mainlabel:
                        break
        else:
            print(f"[DEBUG] no sublabel, text0=<{text0}>")
            print(f"[DEBUG] text_blocks_sorted shape: {text_blocks_sorted.shape}")
            print(f"[DEBUG] text_blocks_sorted columns: {text_blocks_sorted.columns.tolist()}")
            print(f"[DEBUG] text_blocks_sorted=\n{text_blocks_sorted.to_string()}")
            distance_threshold = 10  # distance threshold, adjustable as needed
            
            for idx, row in text_blocks_sorted.iterrows():
                tbbox = row["bbox"]
                text1 = row["text"]
                edge_dist = row["edge_dist"]
                
                print(f"[DEBUG] Checking text block {idx}: bbox={tbbox}, text='{text1[:20]}...', edge_dist={edge_dist:.2f}")
                
                # If too far away, stop searching
                if edge_dist > distance_threshold:
                    print(f"[DEBUG] Text block {idx} too far (edge_dist={edge_dist:.2f} > {distance_threshold}), stopping")
                    break      
                for pat in label_patterns:
                    m = re.search(pat, text1, re.IGNORECASE)
                    if m:
                        mainlabel = m.group(1)
                        mainlabel_bbox = row["bbox"]  # bbox of the text block holding the main label
                        mainlabel_text = row["text"]  # content of the text block holding the main label
                        print(f"[DEBUG] Match: pattern='{pat}', result='{mainlabel}'")
                        break
                    else:
                        print(f"[DEBUG] No match: pattern='{pat}', text='{text1[:30]}...'")
                if mainlabel:
                    break
    
    # If no main label was found, match the nearest text block
    if not mainlabel:
        mainlabel, mainlabel_bbox, mainlabel_text = match_nearest_text_block(text_blocks_sorted, label_dist_threshold)
    
    return mainlabel, sublabel, mainlabel_bbox, mainlabel_text

def match_nearest_text_block(text_blocks_sorted, label_dist_threshold=30):
    """
    When no main label is found, use the nearest text block as the caption.
    """
    mainlabel = None
    mainlabel_bbox = None
    mainlabel_text = None
    
    if len(text_blocks_sorted) > 0 and text_blocks_sorted.iloc[0]["edge_dist"] < label_dist_threshold:
        nearest_row = text_blocks_sorted.iloc[0]
        mainlabel_bbox = nearest_row["bbox"]
        mainlabel_text = nearest_row["text"]
        
        
        print(f"[DEBUG] Using nearest text block as caption: distance={nearest_row['edge_dist']:.2f}, text='{mainlabel_text[:50]}...'")
    
    return mainlabel, mainlabel_bbox, mainlabel_text

def find_context_records(label, key, pdf_name, page_num, xref, path_base):
    """
    Find text blocks mentioning the given label on the current and neighboring PDF pages.
    
    Args:
        label: the label to look for
        key: the image key
        pdf_name: the PDF file name
        page_num: the current page number
        xref: the image reference
        path_base: base path of the extracted PDF data
    
    Returns:
        list: matched records
    """
    context_matched = []
    
    if not label or str(label).strip() == '':
        return context_matched
    
    # Build the search pattern
    label_str = str(label).strip()
    label_str_lower = label_str.lower()
    
    # Extract the numeric part
    m_num = re.search(r"(\d{1,4}[\-–—]\d{1,4}(?:\.\d+)?|\d+(?:\.\d+)?)", label_str_lower)
    if not m_num:
        return context_matched
    
    number = m_num.group(1)
    
    # Build the prefix patterns - only image-related prefixes, excluding Table
    if label_str_lower.startswith("efig") or label_str_lower.startswith("efigure"):
        prefix_patterns = ["efig", "efigure"]
    else:
        prefix_patterns = ["fig", "figure"]
    
    # Build the full search patterns
    search_patterns = []
    for prefix in prefix_patterns:
        # Exact patterns - ensure not preceded by Table; allow any chars after the number
        search_patterns.append(rf"\b(?!table\s+){re.escape(prefix)}[\.\s]*{re.escape(number)}[A-Za-z\s]*\b")
        search_patterns.append(rf"\b(?!table\s+){re.escape(prefix)}[\.\s]*{re.escape(number)}\b")
        # Loose patterns (allow extra chars) - ensure not preceded by Table
        search_patterns.append(rf"(?!table\s+){re.escape(prefix)}[\.\s]*{re.escape(number)}[A-Za-z\s]*")
        search_patterns.append(rf"(?!table\s+){re.escape(prefix)}[\.\s]*{re.escape(number)}")
    
    # Search neighboring pages (current page, +-2 pages)
    for offset in [-2, -1, 0, 1, 2]:
        try:
            page_num_int = int(page_num)
            neighbor_page = f"{page_num_int + offset:05d}"
        except (ValueError, TypeError):
            continue
            
        neighbor_text_path = path_base / pdf_name / neighbor_page / "text.json"
        if not neighbor_text_path.exists():
            continue
            
        try:
            neighbor_text_df = pd.DataFrame(json.load(open(neighbor_text_path)))
        except Exception as e:
            print(f"[WARNING] Failed to read text file {neighbor_text_path}: {e}")
            continue
        
        for _, row in neighbor_text_df.iterrows():
            text_content = row["text"]
            
            # Check whether it matches any search pattern
            matched = False
            for pattern in search_patterns:
                if re.search(pattern, text_content, re.IGNORECASE):
                    matched = True
                    break
            
            if matched:
                # Avoid adding the same context twice
                existing_contexts = [record["context"] for record in context_matched]
                if text_content not in existing_contexts:
                    record = {
                        "image_key": key,
                        "label": label,
                        "context": text_content,
                        "image_pdf_name": row.get("image_pdf_name", pdf_name),
                        "image_page_num": row.get("image_page_num", page_num),
                        "image_xref": row.get("image_xref", xref),
                        "context_page": neighbor_page,
                        "context_page_offset": offset
                    }
                    context_matched.append(record)
    
    return context_matched

def filter_text_blocks(text_info_df, text_include_list, fontsize_range, font_list):
    if text_include_list is not None and len(text_include_list) > 0:
        text_info_df = text_info_df[
            text_info_df["text"].apply(
                lambda x: any(
                    [
                        all([t in x.lower() for t in text_include])
                        for text_include in text_include_list
                    ]
                )
            )
        ]
    if fontsize_range is not None:
        text_info_df = text_info_df[
            text_info_df["size"].apply(
                lambda x: x >= fontsize_range[0] and x <= fontsize_range[1]
            )
        ]
    if font_list is not None and len(font_list) > 0:
        text_info_df = text_info_df[
            text_info_df["font"].apply(lambda x: any([font == x for font in font_list]))
        ]
    return text_info_df

def sort_text_blocks(text_info_df, prioritize_text_under_image):
    if prioritize_text_under_image:
        return text_info_df.sort_values(["edge_dist", "text_under_image"], ascending=[True, False])
    else:
        return text_info_df.sort_values("edge_dist", ascending=True)

# =====================
# Main image-text matching function
# =====================
def match_text(
    path_base,
    key_images,
    text_include_list,
    fontsize_range,
    font_list,
    prioritize_text_under_image,
    return_all=False,
    verbose=False,
):
    """
    For each image key, find the most relevant text block (caption/body) and output the matches.
    Also tries to extract a label via regex and searches the current page and +-2 pages
    for body text that mentions the label.
    """
    pbar = tqdm.tqdm(key_images)

    image_matched = 0  # number of successfully matched images
    image_skipped = 0  # number of skipped images

    text_matched = []  # spatial-distance matching results
    # context_matched = []  # label-based body matching results
    label_patterns = [
        r"(figure\s*\d{1,4}[\-–—]\d{1,4}(?:\.\d+)?)",  # Figure 111-4 or Figure 111-4.1
        r"(fig\.?\s*\d{1,4}[\-–—]\d{1,4}(?:\.\d+)?)",
        r"(efig\.?\s*\d{1,4}[\-–—]\d{1,4}(?:\.\d+)?)",
        r"(efigure\s*\d{1,4}[\-–—]\d{1,4}(?:\.\d+)?)",
        r"(figure\s*\d+(?:\.\d+)?)",
        r"(fig\.?\s*\d+(?:\.\d+)?)",
        r"(efig\.?\s*\d+(?:\.\d+)?)",
        r"(efigure\s*\d+(?:\.\d+)?)"
    ]
    for key in pbar:
        if verbose:
            print(f"\n[match_text] key: {key}")
        pdf_name, page_num, xref = os.path.splitext(key)[0].rsplit("_", 2)
        if verbose:
            print(f"Parsed: pdf_name={pdf_name}, page_num={page_num}, xref={xref}")
        image_json_path = path_base / pdf_name / page_num / "image.json"
        text_json_path = path_base / pdf_name / page_num / "text.json"
        if not image_json_path.exists() or not text_json_path.exists():
            continue
        image_info_df = pd.DataFrame(json.load(open(image_json_path))).T
        image_info_df.index = image_info_df.index.astype(int)
        if int(xref) not in image_info_df.index:
            continue
        image_info = image_info_df.loc[int(xref)]
        text_info_df = pd.DataFrame(json.load(open(text_json_path)))
        text_info_df["image_key"] = key
        text_info_df["image_pdf_name"] = pdf_name
        text_info_df["image_page_num"] = page_num
        text_info_df["image_xref"] = xref

        # 1. Compute spatial relationships
        text_info_df = compute_spatial_features(text_info_df, image_info)
        # Extract the main label and sublabel
        mainlabel, sublabel, mainlabel_bbox, mainlabel_text = extract_label_and_sublabel(text_info_df, label_patterns, label_dist_threshold=30)
        
        # Context lookup uses the main label
        # contexts = find_context_records(mainlabel, key, pdf_name, page_num, xref, path_base)
        # context_matched.extend(contexts)
        # 4. Filter text blocks
        text_info_df = filter_text_blocks(text_info_df, text_include_list, fontsize_range, font_list)
        if len(text_info_df) == 0:
            image_skipped += 1
            continue
        if not isinstance(image_info["bbox"], list):
            image_skipped += 1
            continue
        # 5. Sort text blocks
        text_info = sort_text_blocks(text_info_df, prioritize_text_under_image)
        if return_all:
            for _, row in text_info.iterrows():
                row = row.copy()
                row["mainlabel"] = mainlabel
                row["sublabel"] = sublabel
                row["text"] = mainlabel_text  # content of the text block holding the main label
                row["context"] = None
                row["match_type"] = "spatial"
                text_matched.append(row)
        else:
            if len(text_info) > 0:
                row = text_info.iloc[0].copy()
                row["mainlabel"] = mainlabel
                row["sublabel"] = sublabel
                row["text"] = mainlabel_text  # content of the text block holding the main label
                row["context"] = None
                row["match_type"] = "spatial"
                text_matched.append(row)
        image_matched += 1
    text_matched = pd.DataFrame(text_matched)
    # context_matched = pd.DataFrame(context_matched)
    return text_matched

def refine_matched_pairs(df, path_pdf_extracted):
    df = df.copy().reset_index(drop=True)
    for idx, row in df.iterrows():
        label = row.get("label", None)
        sublabel = row.get("sublabel", None)
        if not label or str(label).strip() == '':
            continue  # do nothing
        pdf_name = row["image_pdf_name"]
        page_num = row["image_page_num"]
        xref = row["image_xref"]
        image_bbox = row["bbox"] if "bbox" in row else None
        text_json_path = path_pdf_extracted / pdf_name / page_num / "text.json"
        if not text_json_path.exists():
            continue
        text_info_df = pd.DataFrame(json.load(open(text_json_path)))
        
        # The data already has an edge_dist field; use it directly
        # Sort by distance, nearest first
        text_blocks_sorted = text_info_df.sort_values("edge_dist", ascending=True)
        
        # Build the label-matching patterns
        label_str = str(label).lower().strip()
        if label_str.startswith("efig") or label_str.startswith("efigure"):
            prefix_patterns = ["efig", "efigure"]
        else:
            prefix_patterns = ["fig", "figure"]
        
        m = re.search(r"(\d{1,4}[\-–—]\d{1,4}(?:\.\d+)?|\d+(?:\.\d+)?)", label_str)
        if m:
            number_core = m.group(1)
        else:
            continue
        
        if not number_core or len(number_core) < 3:
            continue
        
        # Build the matching patterns
        patterns = []
        # Prefer the caption format (start of the text block)
        patterns.append(r"|".join([rf"^\s*{p}[\.\s]*{re.escape(number_core)}\b" for p in prefix_patterns]))
        # Otherwise fall back to matching anywhere
        patterns.append(r"|".join([rf"\b{p}[\.\s]*{re.escape(number_core)}\b" for p in prefix_patterns]))
        
        # Search matching text blocks, nearest first
        matched_text = None
        for _, text_row in text_blocks_sorted.iterrows():
            text_content = text_row["text"].lower()
            
            for pattern in patterns:
                if re.search(pattern, text_content, re.IGNORECASE):
                    matched_text = text_row["text"]
                    break
            
            if matched_text:
                break
        
        # If a matching text was found, update the DataFrame
        if matched_text:
            df.at[idx, "text"] = matched_text
    
    return df



def extract_context_for_images(df, path_pdf_extracted):
    df = df.copy().reset_index(drop=True)
    all_text_blocks = []

    for pdf_name in df['image_pdf_name'].unique():
        pdf_rows = df[df['image_pdf_name'] == pdf_name]
        page_nums = pdf_rows['image_page_num'].unique()
        for page_num in page_nums:
            for offset in [-2, -1, 0, 1, 2]:
                page_num_int = int(page_num) + offset
                page_str = f"{page_num_int:05d}"
                text_json_path = path_pdf_extracted / pdf_name / page_str / "text.json"
                if not text_json_path.exists():
                    continue
                try:
                    with open(text_json_path, encoding='utf-8') as f:
                        text_info = json.load(f)
                    text_info_df = pd.DataFrame(text_info)
                    text_info_df['pdf_name'] = pdf_name
                    text_info_df['page_num'] = page_str
                    all_text_blocks.append(text_info_df)
                except Exception as e:
                    print(f"[WARNING] Failed to load {text_json_path}: {e}")

    if not all_text_blocks:
        df['context'] = [[] for _ in range(len(df))]
        return df

    all_text_blocks = pd.concat(all_text_blocks, ignore_index=True)
    context_list = []

    for idx, row in df.iterrows():
        label = row.get("label", None)
        pdf_name = row['image_pdf_name']
        page_num = row['image_page_num']
        caption_text = str(row.get("text", "")).strip()

        if not label or str(label).strip() == '':
            context_list.append([])
            continue

        label_str = str(label).strip().lower()
        prefix_patterns = ["efig", "efigure", "efig."] if label_str.startswith(('efig', 'efigure', 'efig.')) else ["fig", "figure", "fig."]

        m_num = re.search(r"(\d{1,4}[\-–—]\d{1,4}(?:\.\d+)?|\d+(?:\.\d+)?)", label_str)
        if not m_num:
            context_list.append([])
            continue
        number = m_num.group(1)
        number_escaped = re.escape(number)

        # Only match the main label pattern - ensure not preceded by Table; allow any chars after the number
        pattern = rf"\b(?!table\s+)(?:{'|'.join(prefix_patterns)})[\.\s]*{number_escaped}[A-Za-z\s]*\b"

        matched_blocks = all_text_blocks[
            (all_text_blocks['pdf_name'] == pdf_name) &
            (all_text_blocks['page_num'].astype(int).between(int(page_num)-2, int(page_num)+2)) &
            (all_text_blocks['text'].str.contains(pattern, case=False, regex=True, na=False))
        ]

        matched_blocks = matched_blocks[matched_blocks['text'].str.strip() != caption_text]
        unique_contexts = matched_blocks['text'].drop_duplicates().tolist()
        context_list.append(unique_contexts)

    df['context'] = context_list
    return df

# =====================
# Main entry point
# =====================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="pdf_match.py",
        description="pdf match",
        epilog="",
    )

    parser.add_argument("--image", type=str, help="image path", required=True)
    parser.add_argument("--pdf-extracted", type=str, help="pdf extracted path", required=True)
    parser.add_argument("--config", type=str, help="pdf extracted path", required=True)
    parser.add_argument("-o", "--output", type=str, help="output path", required=True)

    args = parser.parse_args()
    print("Arguments:")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    path_image = Path(args.image)
    path_pdf_extracted = Path(args.pdf_extracted)
    path_output = Path(args.output)
    path_config = Path(args.config)

    # Read the list of image keys (supports hdf5 or pkl)
    if path_image.suffix == ".hdf5":
        key_images = get_hdf5_key(path_input=path_image, field="images")
    elif path_image.suffix == ".pkl":
        key_images = load_pkl(path_input=path_image, field="images")

    # Read the matching config
    with open(path_config) as f:
        config = json.load(f)

    text_matched_all = []
    context_matched_all = []
    for pdf_name, pdf_config_list in config.items():
        for pdf_config in pdf_config_list:
            print(f"\n==== Matching PDF: {pdf_name} config: {pdf_config} ====")
            text_include_list = pdf_config["text_include_list"]
            fontsize_range = pdf_config["fontsize_range"]
            font_list = pdf_config["font_list"]
            prioritize_text_under_image = pdf_config["prioritize_text_under_image"]
            return_all = pdf_config["return_all"]
            # Only process image keys belonging to the current PDF
            key_images_pdf = [
                key
                for key in key_images
                if os.path.splitext(key)[0].rsplit("_", 2)[0] == pdf_name
            ]
            print(f"key_images_pdf count: {len(key_images_pdf)}")
            if len(key_images_pdf) > 0:
                print(f"Example key: {key_images_pdf[0]}")
            else:
                print("[WARNING] No matching image keys!")
            # Call the main matching function
            text_matched = match_text(
                path_base=path_pdf_extracted,
                key_images=key_images_pdf,
                text_include_list=text_include_list,
                fontsize_range=fontsize_range,
                font_list=font_list,
                prioritize_text_under_image=prioritize_text_under_image,
                return_all=return_all,
                verbose=True,
            )
            print(f"text_matched shape this round: {text_matched.shape}")
            text_matched_all.append(text_matched)
            # context_matched_all.append(context_matched)

    # Merge all results
    text_matched_all_df = pd.concat(text_matched_all)
    # context_matched_all_df = pd.concat(context_matched_all)

    # First handle the label/sublabel columns
    for df in [text_matched_all_df]:
        # Build the label column
        if "mainlabel" in df.columns:
            df["label"] = df["mainlabel"]
            df.drop(columns=["mainlabel"], inplace=True)
        else:
            if "label" not in df.columns:
                df["label"] = None
        if "sublabel" not in df.columns:
            df["sublabel"] = None
        # df["label"] = df["label"].where(df["sublabel"].notna(), df["label"])

    # # Then refine
    # text_matched_all_df = refine_matched_pairs(text_matched_all_df, path_pdf_extracted)
    # Extract the global context
    text_matched_all_df = extract_context_for_images(text_matched_all_df, path_pdf_extracted)

    # Output as csv or pkl
    if path_output.suffix == ".csv":
        # Fix the column order so that label and sublabel exist and sublabel follows label
        text_cols = [c for c in text_matched_all_df.columns if c not in ("label", "sublabel")]
        text_matched_all_df[["label", "sublabel"] + text_cols].to_csv(path_output, index=False)
        context_csv_path = str(path_output).replace('.csv', '_context.csv')
        # context_cols = [c for c in context_matched_all_df.columns if c not in ("label", "sublabel")]
        # context_matched_all_df[["label", "sublabel"] + context_cols].to_csv(context_csv_path, index=False)
    elif path_output.suffix == ".pkl":
        text_matched_all_df.to_pickle(path_output)
        context_pkl_path = str(path_output).replace('.pkl', '_context.pkl')
        # context_matched_all_df.to_pickle(context_pkl_path)
    else:
        raise ValueError("output file type not supported")
