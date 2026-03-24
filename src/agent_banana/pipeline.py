from __future__ import annotations

from pathlib import Path
from typing import Tuple

from PIL import Image

from .config import load_dotenv

from .memory import ContextFolder, SessionStore
from .models import PipelineResult, PlanStep, SessionState, StepResult, TurnRecord
from .nano_banana import MockNanoBananaClient, NanoBananaClient, build_image_client
from .planning import RLPlanner, RLValueStore
from .quality import QualityJudge
from .targeting import classify_target, fallback_box_for_profile, max_bbox_area_ratio, refine_bbox_for_profile
from .vision import (
    assess_preview_framing,
    center_box,
    crop_box,
    draw_bbox_overlay,
    encode_png_data_url,
    expand_box,
    fit_image_inside_canvas,
    infer_bbox_from_preview,
    paste_crop,
)


class AgentBananaApp:
    def __init__(
        self,
        *,
        root: Path,
        image_client: NanoBananaClient | None = None,
        max_retries: int = 1,
    ):
        self.root = root
        artifacts_root = self.root / "artifacts" / "agent_banana"
        self.image_client = image_client or build_image_client()
        self.fallback_image_client = MockNanoBananaClient()
        self.context_folder = ContextFolder()
        self.session_store = SessionStore(artifacts_root / "sessions")
        self.planner = RLPlanner(RLValueStore(artifacts_root / "planner_values.json"))
        self.quality_judge = QualityJudge()
        self.max_retries = max_retries

    @classmethod
    def from_env(cls, root: Path | None = None) -> "AgentBananaApp":
        root = root or Path(__file__).resolve().parents[2]
        load_dotenv(root / ".env")
        return cls(root=root)

    def run(self, image: Image.Image, instruction: str, session_id: str | None = None) -> PipelineResult:
        session = self.session_store.load_or_create(session_id)
        current_image = image.convert("RGB")
        folded_context = self.context_folder.fold(session.turns)
        parsed_edits = self.planner.parse_instruction(instruction, folded_context)
        candidate_plans = self.planner.plan(parsed_edits, folded_context)
        selected_plan = candidate_plans[0]

        runtime_mode = self.image_client.mode_label()
        step_results = []
        reward_components = []
        bboxes = []

        for step in selected_plan.steps:
            target_profile = classify_target(step.target, step.verb)
            preview_prompt = self._preview_prompt(instruction, step, folded_context)
            preview_response, preview_mode = self._safe_preview(current_image, preview_prompt)
            if preview_mode != runtime_mode:
                runtime_mode = preview_mode

            normalized_preview, use_preview_for_localization = self._prepare_preview_for_localization(
                current_image,
                preview_response.image,
                step,
                target_profile,
            )
            raw_bbox = None
            if use_preview_for_localization:
                raw_bbox = infer_bbox_from_preview(current_image, normalized_preview, padding=step.padding)
                if raw_bbox is not None and step.scope != "global":
                    image_area = max(1, current_image.size[0] * current_image.size[1])
                    if raw_bbox.area / image_area > max_bbox_area_ratio(target_profile) * 1.8:
                        raw_bbox = None
            if raw_bbox is None:
                if step.scope == "global":
                    bbox = center_box(current_image.size, scale=0.82)
                else:
                    bbox = fallback_box_for_profile(current_image.size, target_profile)
            else:
                bbox = refine_bbox_for_profile(raw_bbox, current_image.size, target_profile)

            composed_image, bbox, quality, attempts, edit_mode = self._apply_step(
                current_image,
                step,
                bbox,
                normalized_preview,
                folded_context.summary,
                target_profile,
            )
            if edit_mode != runtime_mode:
                runtime_mode = edit_mode

            overlay_image = draw_bbox_overlay(current_image, bbox, step.target)
            step_results.append(
                StepResult(
                    step=step,
                    bbox=bbox,
                    quality=quality,
                    preview_data_url=encode_png_data_url(normalized_preview),
                    overlay_data_url=encode_png_data_url(overlay_image),
                    edited_data_url=encode_png_data_url(composed_image),
                    attempts=attempts,
                )
            )
            reward_components.append(quality.score)
            bboxes.append(bbox)
            current_image = composed_image

        reward = 0.0 if not reward_components else sum(reward_components) / len(reward_components)
        self.planner.record_feedback(selected_plan, reward)
        session.turns.append(
            TurnRecord(
                instruction=instruction,
                parsed_edits=parsed_edits,
                selected_plan=selected_plan,
                reward=reward,
                bboxes=bboxes,
            )
        )
        session.folded_context = self.context_folder.fold(session.turns)
        self.session_store.save(session)

        return PipelineResult(
            session_id=session.session_id,
            mode=runtime_mode,
            instruction=instruction,
            folded_context=session.folded_context,
            parsed_edits=parsed_edits,
            candidate_plans=candidate_plans,
            selected_plan=selected_plan,
            source_image=encode_png_data_url(image),
            final_image=encode_png_data_url(current_image),
            step_results=step_results,
            reward=reward,
        )

    def _preview_prompt(self, instruction: str, step: PlanStep, folded_context) -> str:
        target_profile = classify_target(step.target, step.verb)
        profile_note = " Keep the exact same full-image framing, aspect ratio, camera position, and borders. Do not crop, zoom, or reframe the scene."
        if target_profile == "face_accessory" and step.verb == "remove":
            profile_note += (
                " Localize only the eyewear region. Preserve the person's face, skin, eyes, nose, hair, pose, and lighting."
            )
        return (
            f"Session context: {folded_context.summary} "
            f"Current step {step.order}: {step.prompt} "
            f"Global user instruction: {instruction}.{profile_note}"
        )

    def _edit_prompt(self, step: PlanStep, context_summary: str, attempt: int) -> str:
        target_profile = classify_target(step.target, step.verb)
        retry_note = ""
        if attempt > 0:
            retry_note = " Retry with a slightly broader crop and stronger boundary consistency."
        profile_note = " Keep the crop framing fixed and do not zoom or change viewpoint."
        if target_profile == "face_accessory" and step.verb == "remove":
            profile_note += (
                " Remove only the glasses or frames. Keep the same face identity, eyes, eyebrows, nose bridge, skin tone, wrinkles, hair, and head pose unchanged. "
                "Inpaint only the pixels that were occluded by the glasses."
            )
        elif target_profile == "small_accessory" and step.verb == "remove":
            profile_note += " Remove only the accessory and preserve the surrounding object or person."
        return f"Context: {context_summary} Step: {step.prompt}.{profile_note}{retry_note}"

    def _safe_preview(self, image: Image.Image, prompt: str):
        try:
            return self.image_client.generate_preview(image, prompt), self.image_client.mode_label()
        except Exception:
            return self.fallback_image_client.generate_preview(image, prompt), "mock-fallback"

    def _safe_edit(self, crop: Image.Image, prompt: str):
        try:
            return self.image_client.edit_crop(crop, prompt), self.image_client.mode_label()
        except Exception:
            return self.fallback_image_client.edit_crop(crop, prompt), "mock-fallback"

    def _apply_step(
        self,
        current_image: Image.Image,
        step: PlanStep,
        bbox,
        preview_image: Image.Image,
        context_summary: str,
        target_profile: str,
    ) -> Tuple[Image.Image, object, object, int, str]:
        active_box = bbox
        runtime_mode = self.image_client.mode_label()

        for attempt in range(self.max_retries + 1):
            crop = crop_box(current_image, active_box)
            edit_prompt = self._edit_prompt(step, context_summary, attempt)
            edited_response, edit_mode = self._safe_edit(crop, edit_prompt)
            if edit_mode != runtime_mode:
                runtime_mode = edit_mode
            composed = paste_crop(current_image, edited_response.image, active_box)
            quality = self.quality_judge.evaluate(
                current_image,
                composed,
                active_box,
                preview=preview_image,
                target=step.target,
                verb=step.verb,
            )
            if quality.accepted or attempt == self.max_retries:
                return composed, active_box, quality, attempt + 1, runtime_mode
            if target_profile == "face_accessory":
                active_box = refine_bbox_for_profile(expand_box(active_box, 6, current_image.size), current_image.size, target_profile)
            else:
                active_box = expand_box(active_box, max(12, step.padding // 2), current_image.size)

        raise RuntimeError("Unreachable quality loop exit")

    def _prepare_preview_for_localization(
        self,
        current_image: Image.Image,
        preview_image: Image.Image,
        step: PlanStep,
        target_profile: str,
    ) -> tuple[Image.Image, bool]:
        normalized_preview = fit_image_inside_canvas(preview_image, current_image.size)
        assessment = assess_preview_framing(current_image, normalized_preview)
        if step.scope == "global":
            return normalized_preview, True

        # If the preview changes the outer frame too much, treat it as reframed output and
        # fall back to target-aware priors instead of trusting the full-image diff.
        if assessment["average"] > 0.10 or max(assessment["left"], assessment["right"]) > 0.14:
            return normalized_preview, False

        # Small accessory removals are especially sensitive to preview zoom/reframe.
        if target_profile in {"face_accessory", "small_accessory"} and (
            assessment["average"] > 0.03 or max(assessment["left"], assessment["right"]) > 0.05
        ):
            return normalized_preview, False

        return normalized_preview, True
