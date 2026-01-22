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

import asyncio
import datetime
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import omni.kit.app
import omni.log
import omni.usd
from pxr import Gf, Sdf, Tf, Usd, UsdGeom

from .. import backend as be
from ..backend import ai_processing as ai_proc
from .constants import (
    ADV_MODEL_CHOICES,
    DEFAULT_PROMPT1_TEXT,
    DEFAULT_PROMPT2_TEXT,
    IMAGE_DETAIL_CHOICES,
    MODEL_CHOICES,
    REASONING_EFFORT_CHOICES,
    TEXT_VERBOSITY_CHOICES,
    VECTOR_SEARCH_LIMIT,
)


class CommandsMixin:
    def _get_ai_runtime_options(self, base_model_name: str) -> Tuple[str, str, Dict[str, object]]:
        step1_combo = getattr(self, "_ai_step1_model_combo", None)
        step2_combo = getattr(self, "_ai_step2_model_combo", None)
        effort_combo = getattr(self, "_ai_reasoning_effort_combo", None)
        verbosity_combo = getattr(self, "_ai_text_verbosity_combo", None)
        detail_combo = getattr(self, "_ai_image_detail_combo", None)

        step1_index = step1_combo.model.get_item_value_model().as_int if step1_combo else self._ai_step1_model_index
        step2_index = step2_combo.model.get_item_value_model().as_int if step2_combo else self._ai_step2_model_index
        effort_index = effort_combo.model.get_item_value_model().as_int if effort_combo else self._ai_reasoning_effort_index
        verbosity_index = (
            verbosity_combo.model.get_item_value_model().as_int if verbosity_combo else self._ai_text_verbosity_index
        )
        detail_index = detail_combo.model.get_item_value_model().as_int if detail_combo else self._ai_image_detail_index

        step1_model_name = base_model_name
        step2_model_name = base_model_name
        if 0 < step1_index < len(ADV_MODEL_CHOICES):
            step1_model_name = ADV_MODEL_CHOICES[step1_index]
        if 0 < step2_index < len(ADV_MODEL_CHOICES):
            step2_model_name = ADV_MODEL_CHOICES[step2_index]

        reasoning_effort = None
        if 0 < effort_index < len(REASONING_EFFORT_CHOICES):
            reasoning_effort = REASONING_EFFORT_CHOICES[effort_index]

        text_verbosity = None
        if 0 < verbosity_index < len(TEXT_VERBOSITY_CHOICES):
            text_verbosity = TEXT_VERBOSITY_CHOICES[verbosity_index]

        image_detail = None
        if 0 < detail_index < len(IMAGE_DETAIL_CHOICES):
            image_detail = IMAGE_DETAIL_CHOICES[detail_index]

        max_output_tokens = self._ai_max_output_tokens
        if hasattr(self, "_ai_max_output_tokens_model"):
            raw_tokens = self._ai_max_output_tokens_model.as_string.strip()
            if raw_tokens:
                try:
                    max_output_tokens = int(float(raw_tokens))
                except (ValueError, TypeError):
                    max_output_tokens = self._ai_max_output_tokens
        if max_output_tokens <= 0:
            max_output_tokens = None

        max_retries = getattr(self, "_ai_max_retries", 0)
        if hasattr(self, "_ai_max_retries_model"):
            raw_retries = self._ai_max_retries_model.as_string.strip()
            if raw_retries:
                try:
                    max_retries = int(float(raw_retries))
                except (ValueError, TypeError):
                    max_retries = getattr(self, "_ai_max_retries", 0)
        if max_retries < 0:
            max_retries = 0

        retry_delay_sec = getattr(self, "_ai_retry_delay_sec", 0.0)
        if hasattr(self, "_ai_retry_delay_model"):
            raw_delay = self._ai_retry_delay_model.as_string.strip()
            if raw_delay:
                try:
                    retry_delay_sec = float(raw_delay)
                except (ValueError, TypeError):
                    retry_delay_sec = getattr(self, "_ai_retry_delay_sec", 0.0)
        if retry_delay_sec < 0:
            retry_delay_sec = 0.0

        return step1_model_name, step2_model_name, {
            "reasoning_effort": reasoning_effort,
            "max_output_tokens": max_output_tokens,
            "text_verbosity": text_verbosity,
            "image_detail": image_detail,
            "max_retries": max_retries,
            "retry_delay_sec": retry_delay_sec,
        }

    def _get_replacement_history(self, prim_path: str) -> Dict[str, List[str]]:
        if not hasattr(self, "_replacement_history_by_prim") or self._replacement_history_by_prim is None:
            self._replacement_history_by_prim = {}
        history = self._replacement_history_by_prim.get(prim_path)
        if not history:
            history = {"urls": [], "keys": []}
            self._replacement_history_by_prim[prim_path] = history
        return history

    def _append_replacement_history(
        self, prim_path: str, asset_url: Optional[str], identity_key: Optional[str]
    ) -> None:
        limit = int(getattr(self, "_replacement_history_limit", 20))
        if limit <= 0:
            return
        history = self._get_replacement_history(prim_path)
        if asset_url:
            normalized = self._normalize_asset_url(asset_url)
            if normalized and normalized not in history["urls"]:
                history["urls"].append(normalized)
                if len(history["urls"]) > limit:
                    history["urls"] = history["urls"][-limit:]
        if identity_key and identity_key not in history["keys"]:
            history["keys"].append(identity_key)
            if len(history["keys"]) > limit:
                history["keys"] = history["keys"][-limit:]

    async def _select_next_history_url(
        self, history_urls: List[str], current_url: str
    ) -> Optional[str]:
        if not history_urls or len(history_urls) < 2:
            return None
        try:
            start_idx = history_urls.index(current_url)
        except ValueError:
            start_idx = -1

        total = len(history_urls)
        for offset in range(1, total + 1):
            candidate = history_urls[(start_idx + offset) % total]
            if candidate == current_url:
                continue
            if self._is_asset_blacklisted(candidate):
                continue
            candidate_key = await self._get_asset_identity_key(candidate)
            if candidate_key and candidate_key in getattr(self, "_asset_blacklist_keys", set()):
                continue
            return candidate
        return None

    def _write_generation_log(
        self,
        *,
        analysis_text: str,
        prompt1_text: str,
        prompt2_text: str,
        dimensions_text: str,
        image_path: str,
        step1_model_name: str,
        step2_model_name: str,
        ai_overrides: Dict[str, object],
        step1_stats: Dict[str, object],
        step2_stats: Dict[str, object],
        collisions: List[Tuple[str, str, float]],
        layout_json: Dict[str, object],
    ) -> None:
        try:
            log_dir = self._get_log_dir()
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            log_path = os.path.join(log_dir, f"{timestamp}.log")

            reasoning_effort = ai_overrides.get("reasoning_effort") or ai_proc.DEFAULT_REASONING_EFFORT
            text_verbosity = ai_overrides.get("text_verbosity") or ai_proc.DEFAULT_TEXT_VERBOSITY
            image_detail = ai_overrides.get("image_detail") or ai_proc.DEFAULT_IMAGE_DETAIL
            max_output_tokens = ai_overrides.get("max_output_tokens")
            if not isinstance(max_output_tokens, int) or max_output_tokens <= 0:
                max_output_tokens = ai_proc.DEFAULT_MAX_OUTPUT_TOKENS
            max_retries = ai_overrides.get("max_retries")
            if not isinstance(max_retries, int) or max_retries < 0:
                max_retries = 0
            retry_delay_sec = ai_overrides.get("retry_delay_sec")
            if not isinstance(retry_delay_sec, (int, float)) or retry_delay_sec < 0:
                retry_delay_sec = 0.0

            area_size_x = self._extract_optional_float(layout_json, "area_size_X") if isinstance(layout_json, dict) else None
            area_size_y = self._extract_optional_float(layout_json, "area_size_Y") if isinstance(layout_json, dict) else None
            rooms = self._extract_rooms(layout_json) if isinstance(layout_json, dict) else []
            polygon = self._extract_room_polygon(layout_json)
            polygon_bbox = None
            if polygon:
                xs: List[float] = []
                ys: List[float] = []
                for pt in polygon:
                    if isinstance(pt, dict):
                        x_val = self._extract_optional_float_by_keys(pt, ["X", "x"])
                        y_val = self._extract_optional_float_by_keys(pt, ["Y", "y"])
                        if x_val is not None:
                            xs.append(float(x_val))
                        if y_val is not None:
                            ys.append(float(y_val))
                if xs and ys:
                    polygon_bbox = {
                        "min_x": min(xs),
                        "max_x": max(xs),
                        "min_y": min(ys),
                        "max_y": max(ys),
                        "size_x": max(xs) - min(xs),
                        "size_y": max(ys) - min(ys),
                    }

            total_time = float(step1_stats.get("time", 0.0)) + float(step2_stats.get("time", 0.0))
            total_prompt_tokens = int(step1_stats.get("prompt_tokens", 0)) + int(step2_stats.get("prompt_tokens", 0))
            total_completion_tokens = int(step1_stats.get("completion_tokens", 0)) + int(
                step2_stats.get("completion_tokens", 0)
            )
            total_tokens = int(step1_stats.get("total_tokens", 0)) + int(step2_stats.get("total_tokens", 0))

            lines: List[str] = []
            lines.append("AI Layout Generation Log")
            lines.append(f"timestamp: {timestamp}")
            lines.append("")
            lines.append("[AI Settings]")
            lines.append(f"step1_model: {step1_model_name}")
            lines.append(f"step2_model: {step2_model_name}")
            lines.append(f"reasoning_effort: {reasoning_effort}")
            lines.append(f"text_verbosity: {text_verbosity}")
            lines.append(f"image_detail: {image_detail}")
            lines.append(f"max_output_tokens: {max_output_tokens}")
            lines.append(f"max_retries: {max_retries}")
            lines.append(f"retry_delay_sec: {retry_delay_sec}")
            lines.append("")
            lines.append("[Step 1 Stats]")
            lines.append(f"time_sec: {step1_stats.get('time', 0.0)}")
            lines.append(f"input_tokens: {step1_stats.get('prompt_tokens', 0)}")
            lines.append(f"output_tokens: {step1_stats.get('completion_tokens', 0)}")
            lines.append(f"total_tokens: {step1_stats.get('total_tokens', 0)}")
            lines.append("")
            lines.append("[Step 2 Stats]")
            lines.append(f"time_sec: {step2_stats.get('time', 0.0)}")
            lines.append(f"input_tokens: {step2_stats.get('prompt_tokens', 0)}")
            lines.append(f"output_tokens: {step2_stats.get('completion_tokens', 0)}")
            lines.append(f"total_tokens: {step2_stats.get('total_tokens', 0)}")
            lines.append("")
            lines.append("[Total Stats]")
            lines.append(f"time_sec: {total_time}")
            lines.append(f"input_tokens: {total_prompt_tokens}")
            lines.append(f"output_tokens: {total_completion_tokens}")
            lines.append(f"total_tokens: {total_tokens}")
            lines.append("")
            lines.append("[Analysis Text]")
            lines.append(analysis_text or "")
            lines.append("")
            lines.append("[AABB Collisions]")
            if collisions:
                for name1, name2, volume in collisions:
                    lines.append(f"- {name1} vs {name2}: overlap_volume={volume}")
            else:
                lines.append("No collisions.")
            lines.append("")
            lines.append("[Room Size]")
            lines.append(f"area_size_X: {area_size_x}")
            lines.append(f"area_size_Y: {area_size_y}")
            if rooms:
                lines.append(f"room_count: {len(rooms)}")
                for idx, room in enumerate(rooms):
                    room_name = self._extract_room_name(room, idx)
                    room_polygon = self._extract_room_polygon_from_room(room)
                    room_bbox = None
                    if room_polygon:
                        xs: List[float] = []
                        ys: List[float] = []
                        for pt in room_polygon:
                            if isinstance(pt, dict):
                                x_val = self._extract_optional_float_by_keys(pt, ["X", "x"])
                                y_val = self._extract_optional_float_by_keys(pt, ["Y", "y"])
                                if x_val is not None:
                                    xs.append(float(x_val))
                                if y_val is not None:
                                    ys.append(float(y_val))
                        if xs and ys:
                            room_bbox = {
                                "min_x": min(xs),
                                "max_x": max(xs),
                                "min_y": min(ys),
                                "max_y": max(ys),
                                "size_x": max(xs) - min(xs),
                                "size_y": max(ys) - min(ys),
                            }
                    if room_bbox:
                        lines.append(f"room[{idx}] {room_name} bbox: {json.dumps(room_bbox, ensure_ascii=False)}")
                    else:
                        lines.append(f"room[{idx}] {room_name} bbox: None")
            elif polygon_bbox:
                lines.append(f"room_polygon_bbox: {json.dumps(polygon_bbox, ensure_ascii=False)}")
                lines.append(f"room_polygon_points: {json.dumps(polygon, ensure_ascii=False)}")
            lines.append("")
            lines.append("[Inputs]")
            lines.append(f"image_path: {image_path}")
            lines.append("")
            lines.append("[dimensions.txt]")
            lines.append(dimensions_text or "")
            lines.append("")
            lines.append("[prompt1]")
            lines.append(prompt1_text or "")
            lines.append("")
            lines.append("[prompt2]")
            lines.append(prompt2_text or "")

            with open(log_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            omni.log.info(f"AI generation log saved: {log_path}")

        except Exception as exc:
            omni.log.warn(f"Failed to write generation log: {exc}")

    async def _do_ai_generation(self):
        """
        （非同期）AI生成のメインロジック

        このメソッドは asyncio.ensure_future() によって非同期タスクとして実行される。
        Resubmit時には、self._additional_context に追加コンテキストが設定されている。
        """
        try:
            omni.log.info("+"*80)
            omni.log.info("*** _do_ai_generation() STARTED ***")
            omni.log.info("+"*80)
            omni.log.info(f"  Image path: {self._image_path}")
            omni.log.info(f"  Dimensions path: {self._dimensions_path}")
            omni.log.info(f"  Additional context: {len(self._additional_context) if self._additional_context else 0} chars")

            # --- 1. 入力チェック ---
            if not self._image_path:
                omni.log.error("✗ Image file not selected")
                self._analysis_text_model.as_string = "Error: Please select an image file first."
                return

            if not self._dimensions_path:
                omni.log.error("Please select a dimensions file first")
                self._analysis_text_model.as_string = "Error: Please select a dimensions file first."
                return

            # APIキーを取得（起動時に環境変数から読み込み済み）
            api_key = self._api_key_model.as_string.strip()

            if not api_key:
                omni.log.error(
                    "OpenAI API Key is not set. Set the OPENAI_API_KEY environment variable or enter it in the UI."
                )
                self._analysis_text_model.as_string = (
                    "Error: OpenAI API Key is not set. Set the OPENAI_API_KEY environment variable or enter it in the UI."
                )
                return

            # プロンプトとディメンションの読み込み
            prompt1_text, prompt1_source = self._resolve_prompt_text(
                self._prompt1_path, "prompt_1.txt", DEFAULT_PROMPT1_TEXT, "Prompt 1"
            )
            prompt2_text, prompt2_source = self._resolve_prompt_text(
                self._prompt2_path, "prompt_2.txt", DEFAULT_PROMPT2_TEXT, "Prompt 2"
            )
            dimensions_text = self._read_text_with_fallback(self._dimensions_path, "Dimensions", required=True)

            if dimensions_text is None:
                self._analysis_text_model.as_string = "Error: Failed to load dimensions file."
                return

            # モデル名の取得
            model_index = self._model_combo.model.get_item_value_model().as_int
            if model_index < 0 or model_index >= len(MODEL_CHOICES):
                model_index = 0
            model_name = MODEL_CHOICES[model_index]
            step1_model_name, step2_model_name, ai_overrides = self._get_ai_runtime_options(model_name)

            omni.log.info(f"Using Step 1 model: {step1_model_name}")
            omni.log.info(f"Using Step 2 model: {step2_model_name}")
            omni.log.info(f"Image: {self._image_path}")
            omni.log.info(f"Dimensions: {self._dimensions_path}")
            omni.log.info(f"Prompt 1 source: {prompt1_source}")
            omni.log.info(f"Prompt 2 source: {prompt2_source}")
            self._set_ai_status("AI Status: Step 1 (Analyzing image)")

            # 画像のBase64エンコード
            image_base64 = be.encode_image_to_base64(self._image_path)
            if not image_base64:
                omni.log.error("Failed to encode image to base64")
                self._analysis_text_model.as_string = "Error: Failed to encode image to base64."
                return

            if not prompt1_text or not prompt2_text:
                omni.log.error("Failed to load prompt text. Aborting generation.")
                self._analysis_text_model.as_string = "Error: Failed to load prompt text."
                return

            # 修正指示がある場合はprompt1に追加
            if self._additional_context:
                omni.log.info("="*60)
                omni.log.info("【RESUBMISSION WITH ADDITIONAL CONTEXT】")
                omni.log.info(f"Context length: {len(self._additional_context)} characters")
                omni.log.info(f"Context preview: {self._additional_context[:200]}...")
                omni.log.info("="*60)
                prompt1_text += f"\n\n【ユーザーからの追加指示・修正】\n{self._additional_context}"
                # コンテキスト適用後にクリア（次回の実行で混ざらないように）
                # self._additional_context = ""  # これはやらない：承認ステップで使うかもしれない

            # --- 2. ステップ1: 画像分析 (非同期呼び出し) ---
            omni.log.info("=== Step 1: Analyzing image ===")
            analysis_text, step1_stats = await be.step1_analyze_image(
                image_base64,
                prompt1_text,
                dimensions_text,
                step1_model_name,
                api_key,
                **ai_overrides,
            )

            if not analysis_text:
                omni.log.error("Failed to analyze image")
                self._analysis_text_model.as_string = "Error: Failed to analyze image."
                return

            # --- ▼ UIに結果を反映 ▼ ---
            self._analysis_text_model.as_string = f"Step 1 (Analysis) Complete:\n\n{analysis_text}"
            self._set_ai_tokens(step1_stats, None)
            self._set_ai_status("AI Status: Step 1 complete")
            omni.log.info(f"Analysis completed in {step1_stats['time']:.2f}s")
            omni.log.info(f"Analysis result:\n{analysis_text}")

            # 承認モードの場合はここで停止
            if self._require_approval.as_bool:
                omni.log.info("Approval mode enabled. Waiting for user approval...")
                self._analysis_text_model.as_string += (
                    "\n\n⏸️ Waiting for approval. Click 'Approve' to continue or 'Reject' to add context."
                )

                # 分析結果を保存（承認後に使用）
                self._analysis_result = {
                    "analysis_text": analysis_text,
                    "step1_stats": step1_stats,
                    "image_base64": image_base64,
                    "prompt1_text": prompt1_text,
                    "prompt2_text": prompt2_text,
                    "dimensions_text": dimensions_text,
                    "model_name": model_name,
                    "step1_model_name": step1_model_name,
                    "step2_model_name": step2_model_name,
                    "ai_overrides": ai_overrides,
                    "api_key": api_key
                }

                # 承認ボタンを表示
                self._approval_buttons_container.visible = True
                self._approval_pending = True
                self._set_ai_status("AI Status: Waiting for approval")
                self._set_ai_busy(False)
                return  # ここで一旦停止

            # 承認モードがオフの場合は続行
            omni.log.info("Approval mode disabled. Continuing to Step 2...")

            # --- 3. ステップ2: JSON生成 (非同期呼び出し) ---
            omni.log.info("=== Step 2: Generating JSON ===")
            self._analysis_text_model.as_string += "\n\nGenerating... (Step 2: Creating JSON...)"
            self._set_ai_status("AI Status: Step 2 (Generating JSON)")

            layout_json, step2_stats = await be.step2_generate_json(
                analysis_text,
                dimensions_text,
                image_base64,
                prompt2_text,
                step2_model_name,
                api_key,
                **ai_overrides,
            )

            if not layout_json:
                omni.log.error("Failed to generate JSON")
                self._analysis_text_model.as_string += "\nError: Failed to generate JSON."
                return

            self._analysis_text_model.as_string += "\nJSON Generation Complete."
            self._set_ai_tokens(step1_stats, step2_stats)
            self._set_ai_status("AI Status: Step 2 complete")
            omni.log.info(f"JSON generation completed in {step2_stats['time']:.2f}s")

            if self._analysis_mentions_left_positive_x(analysis_text):
                if self._flip_x_axis_in_layout(layout_json):
                    omni.log.warn("Detected left=+X in analysis. Flipped X axis in layout JSON.")

            # JSONをファイルに保存
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
            base_name = os.path.splitext(os.path.basename(self._image_path))[0]
            model_name_safe = step2_model_name.replace("/", "-").replace(":", "-")
            json_filename = f"{base_name}_layout_{model_name_safe}_{timestamp}.json"

            out_dir = self._get_json_output_dir()
            os.makedirs(out_dir, exist_ok=True)
            json_path = os.path.join(out_dir, json_filename)

            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(layout_json, f, indent=4, ensure_ascii=False)
                omni.log.info(f"JSON saved to: {json_path}")
                self._analysis_text_model.as_string += f"\nJSON saved to: {json_path}"
                self._set_generated_json_preview(json_path, layout_json)
            except Exception as e:
                omni.log.error(f"Failed to save JSON: {e}")
                self._analysis_text_model.as_string += f"\nError: Failed to save JSON: {e}"
                return

            # 統計情報の出力
            total_time = step1_stats["time"] + step2_stats["time"]
            total_prompt_tokens = step1_stats["prompt_tokens"] + step2_stats["prompt_tokens"]
            total_completion_tokens = step1_stats["completion_tokens"] + step2_stats["completion_tokens"]
            total_tokens = step1_stats["total_tokens"] + step2_stats["total_tokens"]

            omni.log.info("="*60)
            omni.log.info("【Total Statistics (Step 1 + Step 2)】")
            omni.log.info("="*60)
            omni.log.info(f"Total processing time: {total_time:.2f}s")
            omni.log.info(f"Total prompt tokens: {total_prompt_tokens:,}")
            omni.log.info(f"Total completion tokens: {total_completion_tokens:,}")
            omni.log.info(f"Total tokens: {total_tokens:,}")
            omni.log.info("="*60)

            # --- 4. ステップ3: 衝突検出 ---
            omni.log.info("=== Step 3: Checking collisions ===")
            self._analysis_text_model.as_string += "\n\nChecking collisions..."
            self._set_ai_status("AI Status: Step 3 (Checking collisions)")
            collisions = be.step3_check_collisions(layout_json)

            # --- 5. ステップ4/5: アセット配置 ---
            self._layout_json = layout_json
            self._analysis_text_model.as_string += "\nStarting asset placement..."

            # _start_asset_searchはすでに非同期タスクとして動作
            self._start_asset_search(self._layout_json)

            self._analysis_text_model.as_string += "\n\nProcess finished."
            self._set_ai_status("AI Status: Done (Placement started)")
            omni.log.info("AI JSON generation completed successfully!")

            self._write_generation_log(
                analysis_text=analysis_text,
                prompt1_text=prompt1_text,
                prompt2_text=prompt2_text,
                dimensions_text=dimensions_text,
                image_path=self._image_path,
                step1_model_name=step1_model_name,
                step2_model_name=step2_model_name,
                ai_overrides=ai_overrides,
                step1_stats=step1_stats,
                step2_stats=step2_stats,
                collisions=collisions,
                layout_json=layout_json,
            )

        except asyncio.CancelledError:
            omni.log.warn("AI Generation task was cancelled.")
            self._analysis_text_model.as_string = "Task Cancelled."
            self._set_ai_status("AI Status: Cancelled")
            raise
        except Exception as e:
            omni.log.error(f"AI Generation failed: {e}")
            self._analysis_text_model.as_string = f"Error: {e}"
            self._set_ai_status("AI Status: Error")
        finally:
            self._ai_task = None
            self._set_ai_busy(False)

    async def _do_step2_and_placement(self):
        """Step 2（JSON生成）以降を実行"""
        try:
            if not self._analysis_result:
                omni.log.error("No analysis result available")
                return

            # 保存した分析結果を取得
            analysis_text = self._analysis_result["analysis_text"]
            step1_stats = self._analysis_result["step1_stats"]
            image_base64 = self._analysis_result["image_base64"]
            prompt2_text = self._analysis_result["prompt2_text"]
            dimensions_text = self._analysis_result["dimensions_text"]
            model_name = self._analysis_result["model_name"]
            step2_model_name = self._analysis_result.get("step2_model_name", model_name)
            step1_model_name = self._analysis_result.get("step1_model_name", model_name)
            ai_overrides = self._analysis_result.get("ai_overrides", {})
            api_key = self._analysis_result["api_key"]
            prompt1_text = self._analysis_result.get("prompt1_text", "")

            # --- ステップ2: JSON生成 (非同期呼び出し) ---
            omni.log.info("=== Step 2: Generating JSON ===")
            self._analysis_text_model.as_string += "\n\nGenerating... (Step 2: Creating JSON...)"
            self._set_ai_status("AI Status: Step 2 (Generating JSON)")

            layout_json, step2_stats = await be.step2_generate_json(
                analysis_text,
                dimensions_text,
                image_base64,
                prompt2_text,
                step2_model_name,
                api_key,
                **ai_overrides,
            )

            if not layout_json:
                omni.log.error("Failed to generate JSON")
                self._analysis_text_model.as_string += "\nError: Failed to generate JSON."
                return

            self._analysis_text_model.as_string += "\nJSON Generation Complete."
            self._set_ai_tokens(step1_stats, step2_stats)
            self._set_ai_status("AI Status: Step 2 complete")
            omni.log.info(f"JSON generation completed in {step2_stats['time']:.2f}s")

            if self._analysis_mentions_left_positive_x(analysis_text):
                if self._flip_x_axis_in_layout(layout_json):
                    omni.log.warn("Detected left=+X in analysis. Flipped X axis in layout JSON.")

            # JSONをファイルに保存
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
            base_name = os.path.splitext(os.path.basename(self._image_path))[0]
            model_name_safe = step2_model_name.replace("/", "-").replace(":", "-")
            json_filename = f"{base_name}_layout_{model_name_safe}_{timestamp}.json"

            out_dir = self._get_json_output_dir()
            os.makedirs(out_dir, exist_ok=True)
            json_path = os.path.join(out_dir, json_filename)

            try:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(layout_json, f, indent=4, ensure_ascii=False)
                omni.log.info(f"JSON saved to: {json_path}")
                self._analysis_text_model.as_string += f"\nJSON saved to: {json_path}"
                self._set_generated_json_preview(json_path, layout_json)
            except Exception as e:
                omni.log.error(f"Failed to save JSON: {e}")
                self._analysis_text_model.as_string += f"\nError: Failed to save JSON: {e}"
                return

            # 統計情報の出力
            total_time = step1_stats["time"] + step2_stats["time"]
            total_prompt_tokens = step1_stats["prompt_tokens"] + step2_stats["prompt_tokens"]
            total_completion_tokens = step1_stats["completion_tokens"] + step2_stats["completion_tokens"]
            total_tokens = step1_stats["total_tokens"] + step2_stats["total_tokens"]

            omni.log.info("="*60)
            omni.log.info("【Total Statistics (Step 1 + Step 2)】")
            omni.log.info("="*60)
            omni.log.info(f"Total processing time: {total_time:.2f}s")
            omni.log.info(f"Total prompt tokens: {total_prompt_tokens:,}")
            omni.log.info(f"Total completion tokens: {total_completion_tokens:,}")
            omni.log.info(f"Total tokens: {total_tokens:,}")
            omni.log.info("="*60)

            # --- ステップ3: 衝突検出 ---
            omni.log.info("=== Step 3: Checking collisions ===")
            self._analysis_text_model.as_string += "\n\nChecking collisions..."
            self._set_ai_status("AI Status: Step 3 (Checking collisions)")
            collisions = be.step3_check_collisions(layout_json)

            # --- ステップ4/5: アセット配置 ---
            self._layout_json = layout_json
            self._analysis_text_model.as_string += "\nStarting asset placement..."

            # _start_asset_searchはすでに非同期タスクとして動作
            self._start_asset_search(self._layout_json)

            self._analysis_text_model.as_string += "\n\nProcess finished."
            self._set_ai_status("AI Status: Done (Placement started)")
            omni.log.info("Step 2-5 completed successfully!")

            self._write_generation_log(
                analysis_text=analysis_text,
                prompt1_text=prompt1_text,
                prompt2_text=prompt2_text,
                dimensions_text=dimensions_text,
                image_path=self._image_path,
                step1_model_name=step1_model_name,
                step2_model_name=step2_model_name,
                ai_overrides=ai_overrides,
                step1_stats=step1_stats,
                step2_stats=step2_stats,
                collisions=collisions,
                layout_json=layout_json,
            )

        except asyncio.CancelledError:
            omni.log.warn("Step 2-5 task was cancelled.")
            self._analysis_text_model.as_string += "\nTask Cancelled."
            self._set_ai_status("AI Status: Cancelled")
            raise
        except Exception as e:
            omni.log.error(f"Step 2-5 failed: {e}")
            self._analysis_text_model.as_string += f"\nError: {e}"
            self._set_ai_status("AI Status: Error")
        finally:
            self._ai_task = None
            self._set_ai_busy(False)

    # --- フェーズ4 (共通の開始点) ---
    def _start_asset_search(self, layout_data: dict):
        """JSONデータをもとにUSD Searchを開始する (フェーズ4)"""
        if layout_data is None:
            omni.log.error("No layout data to process.")
            return

        # ステージをZ-Up / メートル単位に設定
        stage = omni.usd.get_context().get_stage()
        if stage:
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            omni.log.info("Stage configured: Z-Up, 1.0 meters per unit")

        omni.log.info("="*50)
        omni.log.info("STARTING PHASE 4: USD Search")
        omni.log.info("Layout data received:")
        omni.log.info(f"{layout_data}")
        omni.log.info("="*50)

        if self._search_task and not self._search_task.done():
            self._search_task.cancel()

        # レイアウトデータを非同期タスク用にコピー
        layout_copy = json.loads(json.dumps(layout_data))
        self._search_task = asyncio.ensure_future(self._search_and_place_assets(layout_copy))

    @staticmethod
    def _looks_like_object_list(value: object) -> bool:
        if not isinstance(value, list):
            return False
        for item in value:
            if not isinstance(item, dict):
                continue
            if any(
                key in item
                for key in (
                    "object_name",
                    "category",
                    "search_prompt",
                    "Length",
                    "Width",
                    "Height",
                    "rotationZ",
                )
            ):
                return True
        return False

    @staticmethod
    def _looks_like_room_list(value: object) -> bool:
        if not isinstance(value, list):
            return False
        for item in value:
            if not isinstance(item, dict):
                continue
            if "room_polygon" in item or "room_name" in item or "room" in item:
                return True
        return False

    def _extract_layout_objects(self, layout_data) -> List[Dict[str, object]]:
        """レイアウトデータから配置対象のオブジェクトリストを抽出する。"""
        if isinstance(layout_data, list):
            return [obj for obj in layout_data if isinstance(obj, dict)]

        if not isinstance(layout_data, dict):
            return []

        candidate_keys = [
            "area_objects_list",
            "objects",
            "items",
            "object_list",
            "layout_objects",
        ]
        for key in candidate_keys:
            value = layout_data.get(key)
            if self._looks_like_object_list(value):
                return [obj for obj in value if isinstance(obj, dict)]

        # 一部のJSONでは layout_data["layout"]["objects"] のようにネストされることがある
        for value in layout_data.values():
            if isinstance(value, list):
                if self._looks_like_room_list(value):
                    continue
                if self._looks_like_object_list(value):
                    return [obj for obj in value if isinstance(obj, dict)]
            elif isinstance(value, dict):
                nested = self._extract_layout_objects(value)
                if nested:
                    return nested

        return []

    def _extract_room_polygon(self, layout_data) -> List[object]:
        """レイアウトデータから部屋外周ポリゴンを抽出する。"""
        if not isinstance(layout_data, dict):
            return []

        candidate_keys = ["room_polygon", "floor_polygon", "room_outline", "polygon", "area_polygon"]
        for key in candidate_keys:
            value = layout_data.get(key)
            if isinstance(value, list):
                return value

        for value in layout_data.values():
            if isinstance(value, dict):
                nested = self._extract_room_polygon(value)
                if nested:
                    return nested
        return []

    def _extract_rooms(self, layout_data) -> List[Dict[str, object]]:
        if not isinstance(layout_data, dict):
            return []
        candidate_keys = ["rooms", "room_list", "areas", "room_objects", "room_data"]
        for key in candidate_keys:
            value = layout_data.get(key)
            if isinstance(value, list):
                rooms = [room for room in value if isinstance(room, dict)]
                if rooms:
                    return rooms
        return []

    def _extract_room_name(self, room_data: Dict[str, object], index: int) -> str:
        name = str(room_data.get("room_name") or room_data.get("name") or room_data.get("area_name") or "").strip()
        return name or f"Room_{index + 1}"

    def _extract_room_polygon_from_room(self, room_data: Dict[str, object]) -> List[object]:
        if not isinstance(room_data, dict):
            return []
        candidate_keys = ["room_polygon", "polygon", "room_outline", "area_polygon", "outline"]
        for key in candidate_keys:
            value = room_data.get(key)
            if isinstance(value, list):
                return value
        return []

    def _extract_room_windows(self, room_data: Dict[str, object]) -> List[Dict[str, object]]:
        if not isinstance(room_data, dict):
            return []
        candidate_keys = ["windows", "window_list", "window_objects", "window_positions"]
        for key in candidate_keys:
            value = room_data.get(key)
            if isinstance(value, list):
                return [obj for obj in value if isinstance(obj, dict)]
        return []

    def _extract_window_objects(self, layout_data) -> List[Dict[str, object]]:
        """レイアウトデータから窓オブジェクト一覧を抽出する。"""
        if not isinstance(layout_data, dict):
            return []

        candidate_keys = ["windows", "window_list", "window_objects", "window_positions"]
        for key in candidate_keys:
            value = layout_data.get(key)
            if isinstance(value, list):
                return [obj for obj in value if isinstance(obj, dict)]

        for value in layout_data.values():
            if isinstance(value, dict):
                nested = self._extract_window_objects(value)
                if nested:
                    return nested
        return []

    @staticmethod
    def _analysis_mentions_left_positive_x(analysis_text: str) -> bool:
        """解析テキストに『左=+X』系の記述があるかを判定する。"""
        if not analysis_text:
            return False
        patterns = [
            r"左\s*[\=＝:はが]\s*\+x",
            r"left\s*(=|is|:)?\s*\+x",
            r"left\s+is\s+positive\s*\+?x",
        ]
        for pattern in patterns:
            if re.search(pattern, analysis_text, re.IGNORECASE):
                return True
        return False

    def _flip_x_axis_in_layout(self, layout_data) -> bool:
        """layout JSONのX座標を反転し、rotationZも左右反転に合わせて補正する。"""
        if layout_data is None:
            return False

        flipped = False

        def _negate(value):
            try:
                return -float(value)
            except (TypeError, ValueError):
                return None

        def _flip_x_entry(entry: Dict[str, object]) -> bool:
            for key in ("X", "x"):
                if key in entry:
                    new_val = _negate(entry.get(key))
                    if new_val is not None:
                        entry[key] = new_val
                        return True
            return False

        def _flip_rotation_entry(entry: Dict[str, object]) -> bool:
            for key in ("rotationZ", "RotationZ", "rotation_z", "Rotation_Z"):
                if key in entry:
                    try:
                        rot = float(entry.get(key))
                    except (TypeError, ValueError):
                        return False
                    entry[key] = (360.0 - rot) % 360.0
                    return True
            return False

        if isinstance(layout_data, list):
            for obj in layout_data:
                if isinstance(obj, dict):
                    if _flip_x_entry(obj):
                        flipped = True
                    if _flip_rotation_entry(obj):
                        flipped = True
        elif isinstance(layout_data, dict):
            if "area_size_X" in layout_data:
                try:
                    layout_data["area_size_X"] = float(layout_data["area_size_X"]) * -1.0
                    flipped = True
                except (TypeError, ValueError):
                    pass

            rooms = self._extract_rooms(layout_data)
            if rooms:
                for room in rooms:
                    objects = self._extract_layout_objects(room)
                    for obj in objects:
                        if not isinstance(obj, dict):
                            continue
                        if _flip_x_entry(obj):
                            flipped = True
                        if _flip_rotation_entry(obj):
                            flipped = True

                    windows = self._extract_room_windows(room)
                    for window in windows:
                        if isinstance(window, dict) and _flip_x_entry(window):
                            flipped = True

                    room_openings = room.get("openings")
                    if isinstance(room_openings, list):
                        for opening in room_openings:
                            if isinstance(opening, dict) and _flip_x_entry(opening):
                                flipped = True

                    polygon = self._extract_room_polygon_from_room(room)
                    if isinstance(polygon, list):
                        for idx, point in enumerate(list(polygon)):
                            if isinstance(point, dict):
                                if _flip_x_entry(point):
                                    flipped = True
                            elif isinstance(point, list) and len(point) >= 2:
                                new_x = _negate(point[0])
                                if new_x is not None:
                                    point[0] = new_x
                                    flipped = True
                            elif isinstance(point, tuple) and len(point) >= 2:
                                new_x = _negate(point[0])
                                if new_x is not None:
                                    polygon[idx] = (new_x, *point[1:])
                                    flipped = True
            else:
                objects = self._extract_layout_objects(layout_data)
                for obj in objects:
                    if not isinstance(obj, dict):
                        continue
                    if _flip_x_entry(obj):
                        flipped = True
                    if _flip_rotation_entry(obj):
                        flipped = True

                windows = self._extract_window_objects(layout_data)
                for window in windows:
                    if isinstance(window, dict) and _flip_x_entry(window):
                        flipped = True

                polygon = self._extract_room_polygon(layout_data)
                if isinstance(polygon, list):
                    for idx, point in enumerate(list(polygon)):
                        if isinstance(point, dict):
                            if _flip_x_entry(point):
                                flipped = True
                        elif isinstance(point, list) and len(point) >= 2:
                            new_x = _negate(point[0])
                            if new_x is not None:
                                point[0] = new_x
                                flipped = True
                        elif isinstance(point, tuple) and len(point) >= 2:
                            new_x = _negate(point[0])
                            if new_x is not None:
                                polygon[idx] = (new_x, *point[1:])
                                flipped = True

            openings = layout_data.get("openings")
            if isinstance(openings, list):
                for opening in openings:
                    if isinstance(opening, dict) and _flip_x_entry(opening):
                        flipped = True

        return flipped

    def _parse_opening_defaults(self, dimensions_text: str) -> Dict[str, float]:
        defaults = {
            "wall_height": 2.5,
            "wall_thickness": 0.1,
            "window_height": 1.2,
            "sill_height": 0.9,
            "window_width": 1.2,
            "door_height": 2.1,
            "door_width": 0.9,
        }

        def _search_value(keys: List[str]) -> Optional[float]:
            if not dimensions_text:
                return None
            for key in keys:
                pattern = re.compile(rf"{key}\s*[:=]\s*([-+]?\d*\.?\d+)", re.IGNORECASE)
                match = pattern.search(dimensions_text)
                if match:
                    try:
                        return float(match.group(1))
                    except (TypeError, ValueError):
                        continue
                pattern = re.compile(rf"{key}\s+([-+]?\d*\.?\d+)", re.IGNORECASE)
                match = pattern.search(dimensions_text)
                if match:
                    try:
                        return float(match.group(1))
                    except (TypeError, ValueError):
                        continue
            return None

        wall_height = _search_value(["wall_height", "wallheight", "ceiling_height", "room_height"])
        if wall_height is not None:
            defaults["wall_height"] = wall_height

        wall_thickness = _search_value(["wall_thickness", "wallthickness"])
        if wall_thickness is not None:
            defaults["wall_thickness"] = wall_thickness

        window_height = _search_value(["window_height", "windowheight", "window_height_m"])
        if window_height is not None:
            defaults["window_height"] = window_height

        sill_height = _search_value(["sill_height", "sillheight", "window_sill_height"])
        if sill_height is not None:
            defaults["sill_height"] = sill_height

        window_width = _search_value(["window_width", "windowwidth"])
        if window_width is not None:
            defaults["window_width"] = window_width

        door_width = _search_value(["door_width", "doorwidth"])
        if door_width is not None:
            defaults["door_width"] = door_width

        return defaults

    def _normalize_opening_object(
        self,
        obj: Dict[str, object],
        opening_type: str,
        defaults: Dict[str, float],
    ) -> Optional[Dict[str, object]]:
        if not isinstance(obj, dict):
            return None

        x = self._extract_float(obj, "X", 0.0)
        y = self._extract_float(obj, "Y", 0.0)
        width_val = self._extract_optional_float_by_keys(obj, ["Width", "width"])
        length_val = self._extract_optional_float_by_keys(obj, ["Length", "length"])
        if width_val is not None and length_val is not None:
            width = max(width_val, length_val)
        elif width_val is not None:
            width = width_val
        elif length_val is not None:
            width = length_val
        else:
            width = defaults.get("window_width" if opening_type == "window" else "door_width", 1.0)

        height = self._extract_optional_float_by_keys(obj, ["Height", "height"])
        if height is None:
            height_key = "window_height" if opening_type == "window" else "door_height"
            height = defaults.get(height_key, 1.0)

        sill = 0.0
        if opening_type == "window":
            sill = self._extract_optional_float_by_keys(obj, ["SillHeight", "sill_height", "Sill", "sill"])
            if sill is None:
                sill = defaults.get("sill_height", 0.0)

        min_width = defaults.get("window_width" if opening_type == "window" else "door_width", 0.5)
        if width < 0.2 and min_width > width:
            width = min_width

        return {
            "type": opening_type,
            "X": x,
            "Y": y,
            "Width": float(width),
            "Height": float(height),
            "SillHeight": float(sill),
        }

    @staticmethod
    def _normalize_polygon_points(points: List[object]) -> List[Tuple[float, float]]:
        normalized: List[Tuple[float, float]] = []
        for entry in points or []:
            if isinstance(entry, dict):
                x = entry.get("X", entry.get("x"))
                y = entry.get("Y", entry.get("y"))
            elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
                x, y = entry[0], entry[1]
            else:
                continue
            try:
                normalized.append((float(x), float(y)))
            except (TypeError, ValueError):
                continue

        if len(normalized) >= 2:
            first = normalized[0]
            last = normalized[-1]
            if abs(first[0] - last[0]) < 1e-6 and abs(first[1] - last[1]) < 1e-6:
                normalized.pop()
        return normalized

    def _collect_room_edges(self, rooms: List[Dict[str, object]]) -> List[Dict[str, object]]:
        edge_map: Dict[Tuple[Tuple[float, float], Tuple[float, float]], Dict[str, object]] = {}

        def _edge_key(p0: Tuple[float, float], p1: Tuple[float, float], precision: int = 3):
            a = (round(p0[0], precision), round(p0[1], precision))
            b = (round(p1[0], precision), round(p1[1], precision))
            return tuple(sorted([a, b]))

        for index, room in enumerate(rooms):
            polygon = self._extract_room_polygon_from_room(room)
            points = self._normalize_polygon_points(polygon)
            if len(points) < 2:
                continue
            for i in range(len(points)):
                p0 = points[i]
                p1 = points[(i + 1) % len(points)]
                if abs(p0[0] - p1[0]) < 1e-6 and abs(p0[1] - p1[1]) < 1e-6:
                    continue
                key = _edge_key(p0, p1)
                if key in edge_map:
                    edge_map[key]["shared"] = True
                    continue
                edge_map[key] = {"start": p0, "end": p1, "shared": False}

        return list(edge_map.values())

    def _get_or_create_room_root(
        self, stage, parent_path: str, room_name: str, index: int
    ) -> str:
        token = self._sanitize_identifier(room_name or f"Room_{index + 1}")
        room_path = f"{parent_path}/{token}"
        if not stage.GetPrimAtPath(room_path):
            UsdGeom.Xform.Define(stage, Sdf.Path(room_path))
        return room_path

    @staticmethod
    def _build_rectangle_polygon(area_size_x: float, area_size_y: float) -> List[Dict[str, float]]:
        half_x = abs(area_size_x) / 2.0
        half_y = abs(area_size_y) / 2.0
        return [
            {"X": -half_x, "Y": -half_y},
            {"X": half_x, "Y": -half_y},
            {"X": half_x, "Y": half_y},
            {"X": -half_x, "Y": half_y},
        ]

    @staticmethod
    def _snap_openings_to_rectangle_edges(
        openings: List[Dict[str, object]],
        half_x: float,
        half_y: float,
    ) -> List[Dict[str, object]]:
        snapped: List[Dict[str, object]] = []
        for opening in openings:
            try:
                x = float(opening.get("X", 0.0))
                y = float(opening.get("Y", 0.0))
            except (TypeError, ValueError):
                continue

            dist_n = abs(y - half_y)
            dist_s = abs(y + half_y)
            dist_e = abs(x - half_x)
            dist_w = abs(x + half_x)
            min_dist = min(dist_n, dist_s, dist_e, dist_w)

            snapped_opening = dict(opening)
            if min_dist == dist_n:
                snapped_opening["Y"] = half_y
            elif min_dist == dist_s:
                snapped_opening["Y"] = -half_y
            elif min_dist == dist_e:
                snapped_opening["X"] = half_x
            else:
                snapped_opening["X"] = -half_x
            snapped.append(snapped_opening)

        return snapped

    async def _search_and_place_assets(self, layout_data) -> None:
        """USD Search を用いてアセット検索と配置を行う非同期タスク。"""
        placed = 0
        skipped = 0

        try:
            dimensions_text = ""
            if getattr(self, "_dimensions_path", ""):
                dimensions_text = self._read_text_with_fallback(
                    self._dimensions_path, "Dimensions", required=False
                ) or ""
            opening_defaults = self._parse_opening_defaults(dimensions_text)

            search_root = self._search_root_model.as_string if self._search_root_model else ""
            search_root = search_root.strip()
            if not search_root:
                omni.log.error("Search root URL is empty. Set the 'Search Root URL' field before running placement.")
                return

            if not search_root.startswith("omniverse://"):
                omni.log.error(f"Search root must start with 'omniverse://'. Current value: {search_root}")
                return

            normalized_root = search_root if search_root.endswith("/") else f"{search_root}/"

            stage = omni.usd.get_context().get_stage()
            if stage is None:
                omni.log.error("No USD stage is available. Open or create a stage before placing assets.")
                return

            area_name = layout_data.get("area_name") if isinstance(layout_data, dict) else None
            root_prim_path = self._get_or_create_root_prim(stage, area_name)

            rooms = self._extract_rooms(layout_data)
            if rooms:
                omni.log.info(f"Detected {len(rooms)} rooms. Running multi-room placement.")
                openings: List[Dict[str, object]] = []
                opening_keys: set = set()
                door_objects: List[Dict[str, object]] = []
                window_objects: List[Dict[str, object]] = []
                room_polygons: List[List[object]] = []

                def _add_opening(opening: Optional[Dict[str, object]]) -> None:
                    if not opening:
                        return
                    try:
                        key = (
                            str(opening.get("type", "")),
                            round(float(opening.get("X", 0.0)), 3),
                            round(float(opening.get("Y", 0.0)), 3),
                            round(float(opening.get("Width", 0.0)), 3),
                            round(float(opening.get("Height", 0.0)), 3),
                            round(float(opening.get("SillHeight", 0.0)), 3),
                        )
                    except (TypeError, ValueError):
                        key = None
                    if key and key in opening_keys:
                        return
                    if key:
                        opening_keys.add(key)
                    openings.append(opening)

                for room_index, room in enumerate(rooms):
                    room_name = self._extract_room_name(room, room_index)
                    room_root_path = self._get_or_create_room_root(
                        stage, root_prim_path, room_name, room_index
                    )
                    room_polygon = self._extract_room_polygon_from_room(room)
                    if not room_polygon:
                        size_x = self._extract_optional_float_by_keys(room, ["area_size_X", "room_size_X", "size_X"])
                        size_y = self._extract_optional_float_by_keys(room, ["area_size_Y", "room_size_Y", "size_Y"])
                        center_x = self._extract_optional_float_by_keys(room, ["center_X", "room_center_X", "centerX"])
                        center_y = self._extract_optional_float_by_keys(room, ["center_Y", "room_center_Y", "centerY"])
                        center = room.get("center") if isinstance(room, dict) else None
                        if isinstance(center, dict):
                            center_x = center_x if center_x is not None else self._extract_optional_float_by_keys(center, ["X", "x"])
                            center_y = center_y if center_y is not None else self._extract_optional_float_by_keys(center, ["Y", "y"])
                        if size_x is not None and size_y is not None:
                            rect = self._build_rectangle_polygon(float(size_x), float(size_y))
                            if center_x is not None or center_y is not None:
                                cx = float(center_x or 0.0)
                                cy = float(center_y or 0.0)
                                rect = [{"X": pt["X"] + cx, "Y": pt["Y"] + cy} for pt in rect]
                            room_polygon = rect
                            room["room_polygon"] = rect
                    if room_polygon:
                        room_polygons.append(room_polygon)

                    room_windows = self._extract_room_windows(room)
                    objects = self._extract_layout_objects(room)
                    if not objects:
                        omni.log.warn(f"Room '{room_name}' has no placeable objects.")

                    floor_generated = False
                    for index, obj in enumerate(objects):
                        name = str(obj.get("object_name", "") or "").strip()
                        category = str(obj.get("category", "") or "").strip()
                        category_lower = category.lower()
                        search_prompt = str(obj.get("search_prompt", "") or "").strip()

                        if not (name or category or search_prompt):
                            omni.log.warn(
                                f"Skipping object #{index + 1} in room '{room_name}': "
                                "missing 'object_name', 'category', and 'search_prompt'."
                            )
                            skipped += 1
                            continue

                        if name.lower() == "floor" or category_lower == "floor":
                            omni.log.info(
                                f"[Procedural] Generating floor for room '{room_name}' ({index + 1}/{len(objects)})"
                            )
                            floor_path = self._create_procedural_floor(stage, room_root_path, obj)
                            if floor_path:
                                omni.log.info(f"Generated floor at {floor_path}")
                                placed += 1
                                floor_generated = True
                            else:
                                skipped += 1
                            continue

                        if category_lower == "door" or "door" in name.lower():
                            door_objects.append(obj)
                            omni.log.info(
                                f"[Door] Detected door '{name}' in room '{room_name}' at X={obj.get('X', 0)}, Y={obj.get('Y', 0)}"
                            )

                        if category_lower == "window" or "window" in name.lower():
                            window_objects.append(obj)
                            omni.log.info(
                                f"[Window] Detected window '{name}' in room '{room_name}' at X={obj.get('X', 0)}, Y={obj.get('Y', 0)}"
                            )
                            continue

                        search_query = self._build_search_query_from_object(obj)
                        omni.log.info(
                            f"[Search] Querying '{search_query}' "
                            f"(Original name: '{name}', category: '{category}') "
                            f"({index + 1}/{len(objects)})"
                        )
                        try:
                            asset_url = await self._semantic_search_asset(search_query, normalized_root)
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:
                            omni.log.error(f"Vector search failed for '{name}': {exc}")
                            skipped += 1
                            continue

                        if not asset_url:
                            omni.log.warn(
                                f"No matching USD asset found for '{name}' (searched as '{search_query}')."
                            )
                            skipped += 1
                            continue

                        prim_path = await self._reference_asset(stage, room_root_path, name, asset_url, obj)
                        if prim_path:
                            omni.log.info(f"Placed '{name}' at {prim_path}")
                            placed += 1
                        else:
                            skipped += 1

                        await asyncio.sleep(0)

                    if room_polygon and not floor_generated:
                        floor_path = self._create_procedural_polygon_floor(stage, room_root_path, room_polygon)
                        if floor_path:
                            placed += 1

                    for win in room_windows:
                        _add_opening(self._normalize_opening_object(win, "window", opening_defaults))

                for door in door_objects:
                    _add_opening(self._normalize_opening_object(door, "door", opening_defaults))
                for window in window_objects:
                    _add_opening(self._normalize_opening_object(window, "window", opening_defaults))

                for opening in layout_data.get("openings", []) if isinstance(layout_data, dict) else []:
                    if not isinstance(opening, dict):
                        continue
                    opening_type = str(opening.get("type", "window") or "window").lower()
                    if opening_type not in ("window", "door"):
                        opening_type = "window"
                    _add_opening(self._normalize_opening_object(opening, opening_type, opening_defaults))

                wall_height = opening_defaults.get("wall_height", 2.5)
                wall_thickness = opening_defaults.get("wall_thickness", 0.1)

                if room_polygons:
                    edges = self._collect_room_edges(rooms)
                    wall_paths, window_paths = self._create_procedural_walls_from_edges(
                        stage,
                        root_prim_path,
                        edges,
                        openings,
                        wall_height,
                        wall_thickness,
                    )
                    placed += len(wall_paths) + len(window_paths)
                    omni.log.info(
                        f"Generated {len(wall_paths)} wall segments and {len(window_paths)} window panes"
                    )
                else:
                    omni.log.warn("No valid room polygons found; skipping wall generation.")

                omni.log.info(f"USD Search placement finished. Placed={placed}, Skipped={skipped}")
                return

            room_polygon = self._extract_room_polygon(layout_data)
            window_objects = self._extract_window_objects(layout_data)

            objects = self._extract_layout_objects(layout_data)
            if not objects:
                omni.log.warn(
                    "Layout data does not contain placeable objects. Expected keys such as "
                    "'area_objects_list', 'objects', or a top-level list of entries."
                )
                return
            omni.log.info(f"Found {len(objects)} candidate objects for placement.")

            # ドアのリストを収集（壁の切り抜き用）
            door_objects = []
            floor_generated = False

            for index, obj in enumerate(objects):
                name = str(obj.get("object_name", "") or "").strip()
                category = str(obj.get("category", "") or "").strip()
                category_lower = category.lower()
                search_prompt = str(obj.get("search_prompt", "") or "").strip()

                if not (name or category or search_prompt):
                    omni.log.warn(f"Skipping object #{index + 1}: missing 'object_name', 'category', and 'search_prompt'.")
                    skipped += 1
                    continue

                # 床の場合は手続き的に生成
                if name.lower() == "floor" or category_lower == "floor":
                    omni.log.info(f"[Procedural] Generating floor ({index + 1}/{len(objects)})")
                    floor_path = self._create_procedural_floor(stage, root_prim_path, obj)
                    if floor_path:
                        omni.log.info(f"Generated floor at {floor_path}")
                        placed += 1
                        floor_generated = True
                    else:
                        skipped += 1
                    continue

                # ドアの場合はリストに追加（後で壁の切り抜きに使用）
                if category_lower == "door" or "door" in name.lower():
                    door_objects.append(obj)
                    omni.log.info(f"[Door] Detected door '{name}' at X={obj.get('X', 0)}, Y={obj.get('Y', 0)}")

                # 窓は別処理（壁開口/ガラス）に回す
                if category_lower == "window" or "window" in name.lower():
                    window_objects.append(obj)
                    omni.log.info(
                        f"[Window] Detected window '{name}' at X={obj.get('X', 0)}, Y={obj.get('Y', 0)}"
                    )
                    continue

                # 検索クエリを生成
                search_query = self._build_search_query_from_object(obj)

                omni.log.info(
                    f"[Search] Querying '{search_query}' "
                    f"(Original name: '{name}', category: '{category}') "
                    f"({index + 1}/{len(objects)})"
                )
                try:
                    # ベクトルベースのセマンティック検索APIを使用
                    # 直接USDファイルのURLが返されるため、再帰的検索は不要
                    asset_url = await self._semantic_search_asset(search_query, normalized_root)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    omni.log.error(f"Vector search failed for '{name}': {exc}")
                    skipped += 1
                    continue

                if not asset_url:
                    omni.log.warn(f"No matching USD asset found for '{name}' (searched as '{search_query}').")
                    skipped += 1
                    continue

                prim_path = await self._reference_asset(stage, root_prim_path, name, asset_url, obj)
                if prim_path:
                    omni.log.info(f"Placed '{name}' at {prim_path}")
                    placed += 1
                else:
                    skipped += 1

                await asyncio.sleep(0)

            # ポリゴン床（床オブジェクトが無い場合の補完）
            if room_polygon and not floor_generated:
                floor_path = self._create_procedural_polygon_floor(stage, root_prim_path, room_polygon)
                if floor_path:
                    placed += 1

            # すべてのオブジェクト配置が完了した後、壁を生成
            omni.log.info("=== Generating procedural walls ===")
            area_size_x = layout_data.get("area_size_X") if isinstance(layout_data, dict) else None
            area_size_y = layout_data.get("area_size_Y") if isinstance(layout_data, dict) else None

            openings: List[Dict[str, object]] = []
            for door in door_objects:
                opening = self._normalize_opening_object(door, "door", opening_defaults)
                if opening:
                    openings.append(opening)
            for window in window_objects:
                opening = self._normalize_opening_object(window, "window", opening_defaults)
                if opening:
                    openings.append(opening)
            for opening in layout_data.get("openings", []) if isinstance(layout_data, dict) else []:
                if not isinstance(opening, dict):
                    continue
                opening_type = str(opening.get("type", "window") or "window").lower()
                if opening_type not in ("window", "door"):
                    opening_type = "window"
                normalized = self._normalize_opening_object(opening, opening_type, opening_defaults)
                if normalized:
                    openings.append(normalized)

            wall_height = opening_defaults.get("wall_height", 2.5)
            wall_thickness = opening_defaults.get("wall_thickness", 0.1)

            if room_polygon:
                wall_paths, window_paths = self._create_procedural_walls_from_polygon(
                    stage,
                    root_prim_path,
                    room_polygon,
                    openings,
                    wall_height,
                    wall_thickness,
                )
                placed += len(wall_paths) + len(window_paths)
                omni.log.info(
                    f"Generated {len(wall_paths)} wall segments and {len(window_paths)} window panes"
                )
            elif isinstance(area_size_x, (int, float)) and isinstance(area_size_y, (int, float)):
                if openings:
                    rect_polygon = self._build_rectangle_polygon(float(area_size_x), float(area_size_y))
                    half_x = abs(float(area_size_x)) / 2.0
                    half_y = abs(float(area_size_y)) / 2.0
                    openings = self._snap_openings_to_rectangle_edges(openings, half_x, half_y)
                    wall_paths, window_paths = self._create_procedural_walls_from_polygon(
                        stage,
                        root_prim_path,
                        rect_polygon,
                        openings,
                        wall_height,
                        wall_thickness,
                    )
                    placed += len(wall_paths) + len(window_paths)
                    omni.log.info(
                        f"Generated {len(wall_paths)} wall segments and {len(window_paths)} window panes"
                    )
                else:
                    wall_paths = self._create_procedural_walls(
                        stage, root_prim_path, float(area_size_x), float(area_size_y), door_objects
                    )
                    placed += len(wall_paths)
                    omni.log.info(f"Generated {len(wall_paths)} wall segments")
            else:
                omni.log.warn(
                    "Cannot generate walls: area_size_X or area_size_Y not found in layout data. "
                    f"area_size_X={area_size_x}, area_size_Y={area_size_y}"
                )

            omni.log.info(f"USD Search placement finished. Placed={placed}, Skipped={skipped}")
        except asyncio.CancelledError:
            omni.log.warn("USD Search placement cancelled.")
            raise
        except Exception as exc:
            omni.log.error(f"Unexpected error during USD Search placement: {exc}")
        finally:
            self._search_task = None

    async def _blacklist_and_replace_selected_asset(self, prim_path: str) -> None:
        await self._replace_selected_asset(prim_path, add_to_blacklist=True)

    async def _replace_selected_asset(self, prim_path: str, add_to_blacklist: bool = False) -> None:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            omni.log.error("No USD stage is available. Open or create a stage before replacing assets.")
            return

        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            omni.log.warn(f"Invalid prim selected: {prim_path}")
            return

        ref_prim = self._get_reference_prim(prim)
        asset_url = self._get_asset_url_from_prim(ref_prim)
        if not asset_url:
            omni.log.warn(f"Selected prim has no asset reference: {prim_path}")
            return
        history_key = str(ref_prim.GetPath())

        identity_key = await self._get_asset_identity_key(asset_url)
        self._append_replacement_history(history_key, asset_url, identity_key)
        if add_to_blacklist:
            added_url = self._add_asset_to_blacklist(asset_url)
            added_key = self._add_asset_key_to_blacklist(identity_key)
            if added_url or added_key:
                self._save_settings_to_json()
                self._update_blacklist_label()
                self._rebuild_blacklist_list()
                omni.log.info(
                    f"Blacklisted asset: {self._normalize_asset_url(asset_url)} (key={identity_key})"
                )
            else:
                omni.log.info(
                    f"Asset already blacklisted: {self._normalize_asset_url(asset_url)} (key={identity_key})"
                )

        data = self._build_data_from_prim_metadata(ref_prim)
        if not data:
            omni.log.warn("Selected prim has no placement metadata. Cannot replace asset in-place.")
            return

        meta = self._get_search_metadata_from_prim(ref_prim)
        search_query = meta.get("search_query") or self._build_search_query_from_object(meta)
        if not search_query:
            omni.log.warn("Unable to determine search query for selected asset.")
            return

        search_root = self._search_root_model.as_string if self._search_root_model else ""
        search_root = search_root.strip()
        if not search_root:
            omni.log.error("Search root URL is empty. Set the 'Search Root URL' field before replacing assets.")
            return
        if not search_root.startswith("omniverse://"):
            omni.log.error(f"Search root must start with 'omniverse://'. Current value: {search_root}")
            return
        normalized_root = search_root if search_root.endswith("/") else f"{search_root}/"

        history = self._get_replacement_history(history_key)
        history_urls_list = [self._normalize_asset_url(u) for u in history.get("urls", []) if u]
        history_urls = {u for u in history_urls_list if u}
        history_keys = {k for k in history.get("keys", []) if k}
        if asset_url:
            history_urls.add(self._normalize_asset_url(asset_url))
        if identity_key:
            history_keys.add(identity_key)

        replacement_url = await self._semantic_search_asset(
            search_query,
            normalized_root,
            exclude_urls=history_urls if history_urls else None,
            exclude_keys=history_keys if history_keys else None,
            min_unique=10,
        )
        if not replacement_url:
            current_norm = self._normalize_asset_url(asset_url) if asset_url else ""
            cycle_url = await self._select_next_history_url(history_urls_list, current_norm)
            if not cycle_url:
                omni.log.warn(
                    f"No replacement asset found for query '{search_query}'. "
                    f"history_urls={len(history_urls)}, history_keys={len(history_keys)}"
                )
                return
            replacement_url = cycle_url
            omni.log.info(
                "No new unique candidates found. Cycling back to previous asset: "
                f"{replacement_url}"
            )

        omni.log.info(
            f"Replacing asset '{asset_url}' with '{replacement_url}' for prim '{ref_prim.GetPath()}'"
        )
        references = ref_prim.GetReferences()
        references.ClearReferences()
        references.AddReference(replacement_url)
        await self._apply_transform(ref_prim, data, replacement_url)
        replacement_key = await self._get_asset_identity_key(replacement_url)
        self._append_replacement_history(history_key, replacement_url, replacement_key)

    def _get_or_create_root_prim(self, stage, area_name: Optional[str]) -> str:
        """配置先のルートXformを取得または作成する。"""
        token = self._sanitize_identifier(area_name or "AssetPlacer")
        root_path = f"/{token}"
        if not stage.GetPrimAtPath(root_path):
            UsdGeom.Xform.Define(stage, Sdf.Path(root_path))
        return root_path

    async def _reference_asset(
        self,
        stage,
        root_prim_path: str,
        object_name: str,
        asset_url: str,
        object_data: Dict[str, object],
    ) -> Optional[str]:
        """アセットを参照として配置し、トランスフォームを適用する。"""
        try:
            child_token = self._sanitize_identifier(object_name or "Asset")
            prim_path = self._get_unique_child_path(stage, root_prim_path, child_token)
            prim = stage.DefinePrim(prim_path, "Xform")
            references = prim.GetReferences()
            references.ClearReferences()
            references.AddReference(asset_url)
            await self._apply_transform(prim, object_data, asset_url)
            return prim_path
        except Exception as exc:
            omni.log.error(f"Failed to reference asset '{asset_url}' for '{object_name}': {exc}")
            return None

    def _get_unique_child_path(self, stage, parent_path: str, child_token: str) -> str:
        """同名Primが存在する場合は連番を付与してユニークなパスを返す。"""
        parent_sdf = Sdf.Path(parent_path)
        candidate = parent_sdf.AppendChild(child_token)
        suffix = 1
        while stage.GetPrimAtPath(candidate):
            candidate = parent_sdf.AppendChild(f"{child_token}_{suffix}")
            suffix += 1
        return str(candidate)

    async def _apply_transform(self, prim, data: Dict[str, object], asset_url: Optional[str] = None) -> None:
        """
        レイアウト情報から取得したトランスフォームをPrimに設定する。

        JSONの寸法（メートル単位）を絶対的な最終寸法として扱い、
        アセットの元のサイズに基づいてスケール比を計算して適用する。
        """
        # アセットのロード完了を待機（最大10フレーム）
        # 参照アセットのジオメトリがロードされるまで、BBoxは正しく計算されない
        max_retries = 10
        time_code = Usd.TimeCode.Default()

        # 参照先のupAxisを取得（キャッシュ付き）。asset_urlが未指定の場合は参照を調査。
        if not asset_url:
            asset_url = self._get_asset_url_from_prim(prim)

        up_axis = self._get_asset_up_axis(asset_url) if asset_url else "Z"
        omni.log.info(f"[UpAxis] Using upAxis='{up_axis}' for prim '{prim.GetPath()}', asset='{asset_url}'")

        for attempt in range(max_retries):
            # 1フレーム待機してUSDがアセットをロードする時間を与える
            await omni.kit.app.get_app().next_update_async()

            # BBoxが有効かチェック
            bbox_cache = UsdGeom.BBoxCache(time_code, ["default"])
            bbox = bbox_cache.ComputeWorldBound(prim)
            bbox_range = bbox.ComputeAlignedRange()
            size_vec = bbox_range.GetMax() - bbox_range.GetMin()

            # サイズが非ゼロ（ジオメトリがロードされた）なら待機完了
            if size_vec[0] > 1e-6 or size_vec[1] > 1e-6 or size_vec[2] > 1e-6:
                omni.log.info(f"Asset geometry loaded after {attempt + 1} frame(s)")
                break
        else:
            # 最大試行回数に達してもロードされない場合は警告
            omni.log.warn(f"Asset geometry may not be fully loaded after {max_retries} frames")

        xformable = UsdGeom.Xformable(prim)

        # ステップ1: 既存トランスフォームのクリア
        try:
            for op in xformable.GetOrderedXformOps():
                prim.RemoveProperty(op.GetAttr().GetName())
            if hasattr(xformable, "ClearXformOpOrder"):
                xformable.ClearXformOpOrder()
            else:
                xformable.SetXformOpOrder([])
        except Exception as exc:  # pragma: no cover - defensive cleanup
            omni.log.warn(f"Failed clearing existing xform ops on '{prim.GetPath()}': {exc}")

        # ステップ2: 「元のサイズ」の計算
        # 待機処理で既に計算したBBoxを再利用
        # （念のため再計算して最新の状態を取得）
        bbox_cache = UsdGeom.BBoxCache(time_code, ["default"])
        bbox = bbox_cache.ComputeWorldBound(prim)
        bbox_range = bbox.ComputeAlignedRange()

        # 元のサイズ（メートル単位）を取得
        original_size_vec = bbox_range.GetMax() - bbox_range.GetMin()
        original_size_x = original_size_vec[0]
        original_size_y = original_size_vec[1]
        original_size_z = original_size_vec[2]

        omni.log.info(
            f"Original asset size (m): X={original_size_x:.4f}, Y={original_size_y:.4f}, Z={original_size_z:.4f}"
        )

        # ステップ3: 回転を先に取得（Z軸周り）
        rotation = self._extract_optional_float_by_keys(
            data,
            [
                "rotationZ",
                "RotationZ",
                "rotation_z",
                "Rotation_Z",
                "rotationY",
                "RotationY",
                "rotation_y",
                "Rotation_Y",
                "rotation",
                "Rotation",
            ],
        )

        base_rotation = rotation if rotation is not None else 0.0
        rotation_offset = self._get_asset_rotation_offset(asset_url)
        effective_rotation = (base_rotation + rotation_offset) % 360.0
        if rotation_offset:
            omni.log.info(
                f"[Transform] rotation base={base_rotation} deg, offset={rotation_offset} deg, "
                f"effective={effective_rotation} deg"
            )

        # ステップ4: 「目標のサイズ」の読み取り（Z-Up座標系）
        # JSONからLength, Height, Width（メートル単位）を読み取り
        json_length = self._extract_float(data, "Length", 0.0)
        json_height = self._extract_float(data, "Height", 0.0)
        json_width = self._extract_float(data, "Width", 0.0)

        # Z-Up座標系: X=左右、Y=奥行き、Z=高さ
        # 回転角度に応じて、LengthとWidthをX/Y軸に割り当て直す
        # 0度/180度: Length→X軸、Width→Y軸（デフォルト）
        # 90度/270度: Length→Y軸、Width→X軸（入れ替え）
        rotation_for_scale = effective_rotation
        if abs(rotation_for_scale - 90) < 45 or abs(rotation_for_scale - 270) < 45:
            # 90度または270度の回転（±45度の範囲）
            target_size_x = json_width   # Width → X軸
            target_size_y = json_length  # Length → Y軸
            omni.log.info(
                "Rotation "
                f"{rotation_for_scale} deg detected: swapping Length/Width mapping (Width->X, Length->Y)"
            )
        else:
            # 0度または180度の回転（デフォルト）
            target_size_x = json_length  # Length → X軸
            target_size_y = json_width   # Width → Y軸

        target_size_z = json_height  # Height → Z軸（常に同じ）

        # ステップ5: 「スケール比」の計算（upAxisに応じてY/Zを入れ替え）
        scale_x, scale_y, scale_z = self._compute_scale_with_up_axis(
            up_axis,
            (original_size_x, original_size_y, original_size_z),
            (target_size_x, target_size_y, target_size_z),
        )
        final_scale = Gf.Vec3f(scale_x, scale_y, scale_z)

        omni.log.info(f"[Transform] up_axis={up_axis}, scale=({scale_x:.4f}, {scale_y:.4f}, {scale_z:.4f})")
        omni.log.info(
            f"[Transform] Target size (m): X={target_size_x:.4f}, Y={target_size_y:.4f}, Z={target_size_z:.4f}"
        )

        # ステップ6: XformOpsの追加（適用順: Scale -> RotateX(必要時) -> RotateZ -> Translate）
        # 追加順は逆（Translate -> RotateZ -> RotateX -> Scale）
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))

        rotate_z_op = xformable.AddRotateZOp()
        rotate_z_op.Set(effective_rotation)

        if up_axis == "Y":
            rotate_x_op = xformable.AddRotateXOp()
            rotate_x_op.Set(90.0)

        scale_op = xformable.AddScaleOp()
        scale_op.Set(final_scale)

        # ステップ7: 回転+スケール後のBBoxを取得し、床合わせのためのTranslate Zを決定
        bbox_cache2 = UsdGeom.BBoxCache(time_code, ["default"])
        bbox2 = bbox_cache2.ComputeWorldBound(prim)
        bbox_range2 = bbox2.ComputeAlignedRange()
        min_z_after_rot_scale = bbox_range2.GetMin()[2]

        # ステップ8: JSON座標を適用（Zは床合わせで上書き）
        x = self._extract_float(data, "X", 0.0)
        y = self._extract_float(data, "Y", 0.0)
        object_name = str(data.get("object_name", "") or "")
        category = str(data.get("category", "") or "")
        search_prompt = str(data.get("search_prompt", "") or "")
        search_query = self._build_search_query_from_object({
            "object_name": object_name,
            "category": category,
            "search_prompt": search_prompt,
        })

        self._store_placement_metadata(
            prim,
            asset_url,
            base_rotation,
            json_length,
            json_width,
            json_height,
            x,
            y,
            object_name=object_name,
            category=category,
            search_prompt=search_prompt,
            search_query=search_query,
        )
        translate_z = -min_z_after_rot_scale
        translate_op.Set(Gf.Vec3d(x, y, translate_z))

        omni.log.info(
            f"[Transform] bbox_min_after_rot_scale={min_z_after_rot_scale:.4f}, "
            f"final_translate=({x:.4f}, {y:.4f}, {translate_z:.4f})"
        )

    def _select_asset_url(
        self,
        response,
        host: str,
        object_name: str,
    ) -> Optional[str]:
        """USD Searchの結果から最適なアセットURLを選択する。"""
        paths = getattr(response, "paths", None)
        if not paths:
            omni.log.warn(f"No paths found in search response for '{object_name}'")
            return None

        normalized_target = self._normalize_name(object_name)
        candidates: List[str] = []
        for path in paths:
            uri = getattr(path, "uri", None)
            if not uri:
                continue
            candidate = f"omniverse://{host}{uri}"
            if not candidate.lower().endswith(".usd"):
                continue
            candidates.append(candidate)

        if not candidates:
            omni.log.warn(f"No USD candidates found for '{object_name}' (searched {len(paths)} paths)")
            return None

        omni.log.info(f"Found {len(candidates)} USD candidates for '{object_name}':")
        for candidate in candidates:
            omni.log.info(f"  - {candidate}")

        # Step 1: 完全一致を試みる
        omni.log.info(f"Trying exact match for normalized target: '{normalized_target}'")
        for candidate in candidates:
            base = candidate.split("/")[-1]
            normalized_base = self._normalize_name(base.split(".")[0])
            omni.log.info(f"  Comparing '{normalized_base}' == '{normalized_target}'")
            if normalized_base == normalized_target:
                omni.log.info(f"  -> Exact match found: {candidate}")
                return candidate

        # Step 2: 部分一致を試みる（クエリの各単語がファイル名と一致するか）
        omni.log.info(f"Trying partial match with query words: {object_name.lower().split()}")
        query_words = object_name.lower().split()
        for candidate in candidates:
            base = candidate.split("/")[-1]
            normalized_base = self._normalize_name(base.split(".")[0])
            for word in query_words:
                normalized_word = self._normalize_name(word)
                omni.log.info(f"  Checking if '{normalized_word}' matches '{normalized_base}'")
                # 単語がファイル名と完全一致、またはファイル名に含まれる場合
                if normalized_word == normalized_base or normalized_word in normalized_base:
                    omni.log.info(f"  -> Partial match found: {candidate}")
                    return candidate

        # Step 3: フォールバック（最初の候補を返す）
        omni.log.info(f"No match found, using first candidate: {candidates[0]}")
        return candidates[0]

    @staticmethod
    def _sanitize_identifier(name: str) -> str:
        token = Tf.MakeValidIdentifier(name) if name else ""
        return token or "Item"

    @staticmethod
    def _normalize_name(value: str) -> str:
        return "".join(ch for ch in value.lower() if ch.isalnum())

    def _extract_float(self, data: Dict[str, object], key: str, default: float = 0.0) -> float:
        value = self._get_value_case_insensitive(data, key)
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            omni.log.warn(f"Invalid numeric value '{value}' for key '{key}'. Using default {default}.")
            return default

    def _extract_optional_float(self, data: Dict[str, object], key: str) -> Optional[float]:
        value = self._get_value_case_insensitive(data, key)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            omni.log.warn(f"Invalid numeric value '{value}' for key '{key}'.")
            return None

    def _extract_optional_float_by_keys(self, data: Dict[str, object], keys: List[str]) -> Optional[float]:
        for key in keys:
            value = self._extract_optional_float(data, key)
            if value is not None:
                return value
        return None

    @staticmethod
    def _get_value_case_insensitive(data: Dict[str, object], key: str) -> Optional[object]:
        if key in data:
            return data[key]
        lowered = key.lower()
        for existing_key, value in data.items():
            if isinstance(existing_key, str) and existing_key.lower() == lowered:
                return value
        return None

    def _create_procedural_floor(
        self,
        stage,
        root_prim_path: str,
        object_data: Dict[str, object],
    ) -> Optional[str]:
        """
        JSONデータから床を手続き的に生成します。

        Args:
            stage: USDステージ
            root_prim_path: 親Primのパス
            object_data: JSONオブジェクトデータ（Length, Width, X, Z を含む）

        Returns:
            生成された床PrimのパスまたはNone
        """
        return self._floor_generator.generate(stage, root_prim_path, object_data)

    def _create_procedural_polygon_floor(
        self,
        stage,
        root_prim_path: str,
        polygon_points: List[object],
    ) -> Optional[str]:
        """ポリゴン外周から床メッシュを生成する。"""
        return self._floor_generator.generate_polygon(stage, root_prim_path, polygon_points)

    def _create_procedural_walls(
        self,
        stage,
        root_prim_path: str,
        area_size_x: float,
        area_size_y: float,
        door_objects: List[Dict[str, object]] = None,
    ) -> List[str]:
        """
        部屋のサイズに基づいて4つの壁を手続き的に生成します。
        ドアの位置がある場合は、その部分を切り抜きます。

        Args:
            stage: USDステージ
            root_prim_path: 親Primのパス
            area_size_x: 部屋のX方向のサイズ（メートル単位）
            area_size_y: 部屋のY方向のサイズ（メートル単位）
            door_objects: ドアオブジェクトのリスト（切り抜き用）

        Returns:
            生成された壁Primのパスのリスト
        """
        return self._wall_generator.generate_walls(
            stage, root_prim_path, area_size_x, area_size_y, door_objects
        )

    def _create_procedural_walls_from_polygon(
        self,
        stage,
        root_prim_path: str,
        polygon_points: List[object],
        openings: Optional[List[Dict[str, object]]] = None,
        wall_height: Optional[float] = None,
        wall_thickness: Optional[float] = None,
    ) -> Tuple[List[str], List[str]]:
        """ポリゴン外周から壁と窓ガラスを生成する。"""
        return self._wall_generator.generate_walls_from_polygon(
            stage, root_prim_path, polygon_points, openings or [], wall_height, wall_thickness
        )

    def _create_procedural_walls_from_edges(
        self,
        stage,
        root_prim_path: str,
        edges: List[Dict[str, object]],
        openings: Optional[List[Dict[str, object]]] = None,
        wall_height: Optional[float] = None,
        wall_thickness: Optional[float] = None,
    ) -> Tuple[List[str], List[str]]:
        """複数部屋のエッジから壁と窓ガラスを生成する。"""
        return self._wall_generator.generate_walls_from_edges(
            stage, root_prim_path, edges, openings or [], wall_height, wall_thickness
        )

    def _normalize_furniture_name(self, json_name: str) -> str:
        """
        AIが生成したJSONの家具名を、USD Searchが理解できる
        正規化された検索クエリに変換します。
        """

        # 1. 既知の名前をマッピング
        #    (この辞書は必要に応じて拡張してください)
        name_mapping = {
            # ベッド
            "accessible_master_bed": "bed",
            "left_side_access_bed": "bed",
            "master_bed": "bed",
            # 冷蔵庫
            "compact_refrigerator": "refrigerator",
            "tall_refrigerator": "refrigerator",
            # 椅子
            "support_chair": "chair",
            "office_chair": "chair",
            "upper_corner_chair": "chair",
            "transfer_chair": "chair",
            # テレビ
            "wall_mounted_tv": "television",
            "left_wall_wallmount_tv": "television",
            "slim_wall_tv": "television",
            "wall_tv": "television",
            # 引き出し・収納
            "accessible_drawer": "drawer",
            "lower_corner_drawer": "drawer",
            "low_drawer": "drawer",
            # ドア
            "main_entrance_door": "door",
            "entrance_door": "door",
            "inward_swing_door": "door",
            "swing_door": "door",
            # その他
            "office_table_1": "office table",
            "office_cabinet": "cabinet",
            "office_lamp": "lamp",
            "office_printer": "printer",
            "single_spiral_notepad": "notepad",
        }

        normalized_name = json_name.lower().replace("_", " ")

        # マッピング辞書（キーを小文字化・スペース化して）で検索
        for key, value in name_mapping.items():
            if key.replace("_", " ") in normalized_name:
                omni.log.info(f"Normalized '{json_name}' -> '{value}' (using mapping)")
                return value

        # 2. マッピングがない場合のフォールバック
        #    アンダースコアを除去して最後の単語を使用
        words = normalized_name.split()
        if words:
            fallback_query = words[-1]
            omni.log.info(f"Normalized '{json_name}' -> '{fallback_query}' (using fallback)")
            return fallback_query

        omni.log.warn(f"Could not normalize '{json_name}', using original.")
        return json_name

    def _build_search_query_from_object(self, obj: Dict[str, object]) -> str:
        """
        JSONオブジェクト1件からセマンティック検索用のクエリ文字列を組み立てる。

        優先順:
        1) search_prompt があればそれをベースに使用
        2) search_prompt が空で object_name があれば _normalize_furniture_name で正規化
        3) どちらも無ければ汎用プレースホルダを使う
        category があれば先頭に付与する。
        """
        object_name = str(obj.get("object_name", "") or "").strip()
        category = str(obj.get("category", "") or "").strip()
        search_prompt = str(obj.get("search_prompt", "") or "").strip()

        if search_prompt:
            base_text = search_prompt
        elif object_name:
            base_text = self._normalize_furniture_name(object_name)
        else:
            base_text = "generic furniture object"

        if category:
            query = f"{category}, {base_text}"
        else:
            query = base_text

        omni.log.info(
            f"[SearchQuery] category='{category}', name='{object_name}', "
            f"search_prompt_len={len(search_prompt)}, query='{query}'"
        )
        return query

    @staticmethod
    def _compute_scale_with_up_axis(
        up_axis: str,
        original_sizes: Tuple[float, float, float],
        target_sizes: Tuple[float, float, float],
        min_size: float = 1e-6,
    ) -> Tuple[float, float, float]:
        """
        upAxisに応じたスケール比を計算する。

        Z-upステージでの想定:
        - Z-upアセット: (X→X, Y→Y, Z→Z)
        - Y-upアセット: RotateX(+90) を適用してZ-upに補正するため、(X→X, Z→Y, Y→Z) を考慮
        """
        original_x, original_y, original_z = original_sizes
        target_x, target_y, target_z = target_sizes

        def safe_div(num: float, denom: float) -> float:
            if abs(denom) > min_size and num > 0:
                return num / denom
            return 1.0

        if up_axis == "Y":
            # RotateX(+90) により local Y -> world Z, local Z -> world Y
            scale_x = safe_div(target_x, original_x)
            scale_y = safe_div(target_z, original_y)
            scale_z = safe_div(target_y, original_z)
        else:
            scale_x = safe_div(target_x, original_x)
            scale_y = safe_div(target_y, original_y)
            scale_z = safe_div(target_z, original_z)

        return scale_x, scale_y, scale_z

    def _get_asset_up_axis(self, asset_url: str) -> str:
        """
        アセットのupAxisを取得し、キャッシュする。未知・取得失敗時は'Y'を返す。
        """
        if not asset_url:
            omni.log.warn("[UpAxis] asset_url is empty. Fallback to 'Y'.")
            return "Y"

        if not hasattr(self, "_asset_up_axis_cache"):
            self._asset_up_axis_cache = {}

        cached = self._asset_up_axis_cache.get(asset_url)
        if cached:
            omni.log.info(f"[UpAxis] Cache hit: {asset_url} -> {cached}")
            return cached

        up_axis = "Y"
        try:
            stage = Usd.Stage.Open(asset_url)
            if not stage:
                omni.log.warn(f"[UpAxis] Failed to open stage for {asset_url}. Fallback to 'Y'.")
            else:
                axis_token = UsdGeom.GetStageUpAxis(stage)
                up_axis = "Z" if axis_token == UsdGeom.Tokens.z else "Y"
                omni.log.info(f"[UpAxis] Retrieved upAxis='{up_axis}' for {asset_url}")
        except Exception as exc:
            omni.log.warn(f"[UpAxis] Error while getting upAxis for {asset_url}: {exc}. Fallback to 'Y'.")

        self._asset_up_axis_cache[asset_url] = up_axis
        return up_axis

    async def _get_asset_size(self, asset_url: str) -> Optional[int]:
        if not asset_url:
            return None
        normalized = self._normalize_asset_url(asset_url)
        cache = getattr(self, "_asset_size_cache", None)
        if cache is None:
            self._asset_size_cache = {}
            cache = self._asset_size_cache
        if normalized in cache:
            return cache[normalized]
        try:
            import omni.client
            result, entry = await omni.client.stat_async(normalized)
            if result == omni.client.Result.OK and entry:
                size = int(entry.size) if entry.size is not None else None
            else:
                size = None
        except Exception as exc:
            omni.log.warn(f"[AssetSize] Failed to stat '{normalized}': {exc}")
            size = None
        cache[normalized] = size
        return size

    async def _get_asset_identity_key(
        self, asset_url: str, size_bytes: Optional[int] = None
    ) -> Optional[str]:
        if not asset_url:
            return None
        if size_bytes is None:
            size_bytes = await self._get_asset_size(asset_url)
        return self._build_asset_identity_key(asset_url, size_bytes)

    async def _semantic_search_asset(
        self,
        query: str,
        search_root: str,
        exclude_urls: Optional[set] = None,
        exclude_keys: Optional[set] = None,
        min_unique: int = 1,
        max_limit: int = 200,
    ) -> Optional[str]:
        """
        外部セマンティック検索API (Vector Search) を使用してアセットを検索します。
        GETではなくPOSTを使用し、CLIP埋め込みベクトルによる直接検索を行います。

        Args:
            query: 検索クエリ (例: "A comfortable chair")
            search_root: 検索ルートURL（今回は使用しないが互換性のため残す）
        Returns:
            アセットのOmniverse URL、または見つからない場合はNone
        """
        import requests
        import asyncio
        import json

        # エンドポイント設定
        api_url = "http://192.168.11.65:30080/search"
        api_basic_auth = "omniverse:tsukuverse"

        try:
            # 同期処理を別スレッドで実行（UIをブロックしない）
            loop = asyncio.get_event_loop()
            desired_unique = max(1, int(min_unique or 1))
            limit = max(VECTOR_SEARCH_LIMIT, desired_unique)
            max_limit = max(limit, int(max_limit or limit))
            prev_total_hits = -1
            best_candidate = None

            def sync_request(payload):
                user, password = api_basic_auth.split(":", 1)
                response = requests.post(api_url, json=payload, auth=(user, password), timeout=60)
                response.raise_for_status()
                return response.json()

            usd_extensions = (".usd", ".usda", ".usdc", ".usdz")
            excluded_urls = set()
            for url in getattr(self, "_asset_blacklist", set()):
                if url:
                    excluded_urls.add(self._normalize_asset_url(str(url)))
            if exclude_urls:
                for url in exclude_urls:
                    if url:
                        excluded_urls.add(self._normalize_asset_url(str(url)))

            excluded_keys = set(getattr(self, "_asset_blacklist_keys", set()))
            if exclude_keys:
                for key in exclude_keys:
                    if key:
                        excluded_keys.add(str(key))

            def _coerce_size(value) -> Optional[int]:
                try:
                    size_val = int(value)
                    return size_val if size_val >= 0 else None
                except (TypeError, ValueError):
                    return None

            def _extract_size(data: object) -> Optional[int]:
                if not isinstance(data, dict):
                    return None
                for key in ("size", "file_size", "fileSize", "bytes", "file_size_bytes", "content_length"):
                    if key in data:
                        size_val = _coerce_size(data.get(key))
                        if size_val is not None:
                            return size_val
                nested = data.get("stat")
                if isinstance(nested, dict):
                    for key in ("size", "file_size", "bytes"):
                        if key in nested:
                            size_val = _coerce_size(nested.get(key))
                            if size_val is not None:
                                return size_val
                return None
            while True:
                payload = {
                    "vector_queries": [
                        {
                            "field_name": "clip-embedding.embedding",
                            "query_type": "text",
                            "query": query,
                        }
                    ],
                    "return_metadata": True,
                    "return_images": False,
                    "limit": limit,
                    "file_extension_include": "usd,usda,usdc,usdz",
                    "file_extension_exclude": "png,jpg,jpeg",
                }

                omni.log.info(f"[Vector Search] POST Request to: {api_url}")
                omni.log.info(f"  Payload: {json.dumps(payload, ensure_ascii=False)}")

                result = await loop.run_in_executor(None, lambda: sync_request(payload))

                omni.log.info("[Vector Search] API Response received")
                omni.log.info(f"  Response: {json.dumps(result, ensure_ascii=False)[:500]}")

                # --- レスポンスからUSDパスを抽出 ---
                items = []
                if isinstance(result, list):
                    items = result
                elif isinstance(result, dict):
                    if "results" in result:
                        items = result["results"]
                    elif "hits" in result:
                        if isinstance(result["hits"], dict) and "hits" in result["hits"]:
                            items = result["hits"]["hits"]
                    elif "items" in result:
                        items = result["items"]

                if not items:
                    omni.log.warn(f"[Vector Search] No results found for '{query}'")
                    return best_candidate

                candidates = []
                for item in items:
                    item_data = item.get("_source", item) if isinstance(item, dict) else {}
                    candidate_keys = ["uri", "url", "path", "file_path", "asset_path"]
                    size_val = _extract_size(item_data) or _extract_size(item)
                    for key in candidate_keys:
                        if isinstance(item_data, dict) and key in item_data:
                            val = item_data[key]
                            if val and isinstance(val, str):
                                if val.lower().endswith(usd_extensions):
                                    candidates.append((val, size_val))
                                    break
                                omni.log.info(f"[Vector Search] Skipping non-USD file: {val}")

                if not candidates:
                    omni.log.warn(f"[Vector Search] No USD files found in results for '{query}'")
                    return best_candidate

                seen_keys = set()
                unique_candidates = []
                for candidate_url, candidate_size in candidates:
                    normalized_url = self._normalize_asset_url(candidate_url)
                    if normalized_url in excluded_urls:
                        omni.log.info(f"[Vector Search] Skipping blacklisted asset: {normalized_url}")
                        continue

                    size_bytes = candidate_size
                    if size_bytes is None:
                        size_bytes = await self._get_asset_size(normalized_url)

                    identity_key = self._build_asset_identity_key(normalized_url, size_bytes)
                    if identity_key and identity_key in excluded_keys:
                        omni.log.info(
                            f"[Vector Search] Skipping blacklisted key: {identity_key} ({normalized_url})"
                        )
                        continue
                    if identity_key and identity_key in seen_keys:
                        continue
                    if identity_key:
                        seen_keys.add(identity_key)

                    unique_candidates.append(normalized_url)

                if unique_candidates:
                    best_candidate = unique_candidates[0]

                if len(unique_candidates) >= desired_unique:
                    omni.log.info(f"[Vector Search] Final Asset URL: {unique_candidates[0]}")
                    return unique_candidates[0]

                total_hits = len(items)
                if limit >= max_limit or total_hits <= prev_total_hits:
                    if best_candidate:
                        omni.log.info(f"[Vector Search] Final Asset URL: {best_candidate}")
                    else:
                        omni.log.warn(f"[Vector Search] No USD files found after filtering for '{query}'")
                    return best_candidate

                prev_total_hits = total_hits
                limit = min(limit * 2, max_limit)

        except requests.exceptions.RequestException as exc:
            omni.log.error(f"[Vector Search] HTTP request failed: {exc}")
            return None
        except Exception as exc:
            omni.log.error(f"[Vector Search] Unexpected error: {exc}")
            import traceback
            omni.log.error(f"Stack trace:\n{traceback.format_exc()}")
            return None