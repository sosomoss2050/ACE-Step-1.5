"""Tests for external LM caption and metadata helpers."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from acestep.text_tasks.external_lm_captioning import (
    apply_user_metadata_overrides,
    build_fallback_caption,
    build_format_request_intent,
    caption_needs_retry,
)


class ExternalLmCaptioningTests(unittest.TestCase):
    """Verify caption formatting helpers stay deterministic and local."""

    def test_caption_needs_retry_for_unchanged_or_too_short_result(self) -> None:
        """Simple echoes and very short captions should trigger one retry."""

        self.assertTrue(
            caption_needs_retry(
                original_caption="Salsa dura with brass section, call-and-response vocals",
                generated_caption="Salsa dura with brass section, call-and-response vocals",
            )
        )
        self.assertTrue(
            caption_needs_retry(
                original_caption="Progressive trance instrumental",
                generated_caption="Progressive trance instrumental with pads",
            )
        )
        self.assertFalse(
            caption_needs_retry(
                original_caption="Progressive trance instrumental",
                generated_caption=(
                    "A progressive trance instrumental opens with evolving pads and "
                    "arpeggiators, builds through a long breakdown, and resolves in a "
                    "euphoric outro."
                ),
            )
        )

    def test_apply_user_metadata_overrides_preserves_constrained_values(self) -> None:
        """User-supplied metadata should win over provider drift."""

        plan = SimpleNamespace(
            bpm=1,
            duration=2.4,
            keyscale="C minor",
            timesignature="3/4",
            language="English",
        )

        result = apply_user_metadata_overrides(
            plan=plan,
            user_metadata={
                "bpm": 125,
                "duration": 240,
                "keyscale": "D major",
                "timesignature": "4/4",
                "language": "es",
            },
        )

        self.assertEqual(result.bpm, 125)
        self.assertEqual(result.duration, 240.0)
        self.assertEqual(result.keyscale, "D major")
        self.assertEqual(result.key_scale, "D major")
        self.assertEqual(result.timesignature, "4/4")
        self.assertEqual(result.time_signature, "4/4")
        self.assertEqual(result.language, "es")
        self.assertEqual(result.vocal_language, "es")

    def test_build_fallback_caption_uses_prompt_and_metadata(self) -> None:
        """Fallback caption should expand the original prompt into a narrative."""

        caption = build_fallback_caption(
            caption="Salsa dura with brass section, call-and-response vocals, live club energy",
            user_metadata={
                "bpm": 125,
                "duration": 240,
                "keyscale": "D major",
                "timesignature": "4/4",
            },
        )

        self.assertIn("Salsa dura with brass section", caption)
        self.assertIn("125 BPM", caption)
        self.assertIn("4/4", caption)
        self.assertIn("D major", caption)
        self.assertIn("240 seconds", caption)

    def test_build_format_request_intent_omits_unknown_metadata(self) -> None:
        """Unknown metadata values should not be emitted into the request intent."""

        intent = build_format_request_intent(
            caption="Dreamy synth-pop",
            lyrics="City lights / carry me home",
            user_metadata={
                "bpm": 118,
                "duration": "",
                "keyscale": "C Major",
                "timesignature": "4/4",
                "language": "unknown",
            },
        )

        self.assertIn("Caption: Dreamy synth-pop", intent)
        self.assertIn("Lyrics: City lights / carry me home", intent)
        self.assertIn("bpm: 118", intent)
        self.assertIn("keyscale: C Major", intent)
        self.assertIn("timesignature: 4/4", intent)
        self.assertNotIn("language: unknown", intent)
        self.assertNotIn("duration:", intent)


if __name__ == "__main__":
    unittest.main()
