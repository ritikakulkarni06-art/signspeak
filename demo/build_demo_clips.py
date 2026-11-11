"""Fetch the recorded clips for the demo app's Demo Clip mode into demo/clips/.

Run from the project root:
    python -m demo.build_demo_clips

demo/clips/clips.json lists the clips: ASL Citizen test-split videos, so from
signers the recognizer never trained on, one sign and one signer each. The
videos themselves are gitignored (ASL Citizen may not be redistributed), so
this script streams just those files out of the official archive, as
build_replay_video.py does.

How they were chosen (recorded per clip in clips.json):
  * signs with high held-out accuracy, avoiding `where` and `four`, which the
    live audit found unreliable at low frame rates;
  * the clip classified correctly, with high confidence, from its Stage 3
    features (`test_confidence`);
  * then played through tracking, segmentation and recognition at 4 to 15 fps,
    as Demo Clip mode plays it (`playback_check`): kept only if every run
    produced exactly the right sign and nothing else. Short clips (about 1 s)
    were the ones that failed at the low rates this laptop actually runs at.
That is selection of which recordings to show, not tuning: the clips go through
the unchanged models, and nothing in the app knows their labels.
"""

from __future__ import annotations

import json
from pathlib import Path

from data.prepare_dataset import ASL_CITIZEN_URL
from data.remote_zip import RemoteZip

CLIPS = Path(__file__).resolve().parent / "clips"


def fetch(clips_dir: Path = CLIPS, archive: RemoteZip | None = None) -> list[Path]:
    clips = json.loads((clips_dir / "clips.json").read_text(encoding="utf-8"))["clips"]
    fetched = []
    for clip in clips:
        path = clips_dir / clip["file"]
        if path.is_file():
            print(f"  have  {clip['file']}")
            continue
        archive = archive or RemoteZip(ASL_CITIZEN_URL)
        path.write_bytes(archive.read(f"ASL_Citizen/videos/{clip['file']}"))
        fetched.append(path)
        print(f"  got   {clip['file']}  ({clip['label']}, signer {clip['signer']})")
    print(f"{len(clips)} demo clips in {clips_dir}")
    return fetched


if __name__ == "__main__":
    fetch()
