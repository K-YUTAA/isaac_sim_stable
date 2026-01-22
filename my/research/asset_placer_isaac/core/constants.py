# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

DEFAULT_PROMPT1_TEXT = (
    "You are an interior layout analyst. Provide a concise summary for layout generation.\n"
    "- Coordinate system: X = right (left is -X), Y = depth (up in the image), Z = up.\n"
    "- Origin: lower-left inner corner of the room.\n"
    "- Units: meters.\n"
    "- Output format: 6-12 bullet lines max. Each line: object name, count, "
    "approx center (x,y), approx size (LxW), and orientation note if obvious.\n"
    "- If multiple rooms exist, name each room and describe its boundary/outline and openings.\n"
    "- No long explanations, no headings, no tables."
)

DEFAULT_PROMPT2_TEXT = (
    "Create a JSON layout description for the room(s) based on the analysis, dimensions text, and floor plan image. "
    "Output MUST be valid JSON only (no prose).\n"
    "Schema (multi-room preferred):\n"
    "{\n"
    '  "house_name": string,\n'
    '  "area_size_X": number,  # meters (optional reference)\n'
    '  "area_size_Y": number,  # meters (optional reference)\n'
    '  "rooms": [\n'
    "    {\n"
    '      "room_name": string,\n'
    '      "room_type": string (optional),\n'
    '      "room_polygon": [ { "X": number, "Y": number }, ... ],\n'
    '      "windows": [ { "X": number, "Y": number, "Width": number, "Height": number, "SillHeight": number } ],\n'
    '      "area_objects_list": [\n'
    "        {\n"
    '          "object_name": string,\n'
    '          "category": string (optional),\n'
    '          "search_prompt": string (optional),\n'
    '          "X": number,\n'
    '          "Y": number,\n'
    '          "Length": number,\n'
    '          "Width": number,\n'
    '          "Height": number,\n'
    '          "rotationZ": number\n'
    "        }\n"
    "      ]\n"
    "    }\n"
    "  ],\n"
    '  "openings": [  # optional doors/windows across all rooms\n'
    '    { "type": "door|window", "X": number, "Y": number, "Width": number, "Height": number, "SillHeight": number }\n'
    "  ]\n"
    "}\n"
    "Legacy single-room schema is still accepted if only one room exists. "
    "Notes: Z is up; positions are X/Y only and are global across all rooms. "
    "X is right (+X), left is -X. Use meters everywhere."
)

# UI表示/実呼び出しで共通に使うモデル一覧（ComboBoxの順序 = 保存されるmodel_index）
MODEL_CHOICES = [
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-5-mini",
    "gpt-5",
    "gpt-5.1",
    "gpt-5.2",
]

ADV_MODEL_CHOICES = ["(Use main model)"] + MODEL_CHOICES + ["gpt-5.2-pro"]
REASONING_EFFORT_CHOICES = ["(default)", "low", "medium", "high", "xhigh"]
TEXT_VERBOSITY_CHOICES = ["(default)", "low", "medium", "high"]
IMAGE_DETAIL_CHOICES = ["(default)", "low", "high"]

# Vector Searchで取得する候補数の上限
VECTOR_SEARCH_LIMIT = 50
