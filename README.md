# SignSpeak

Real-time sign-language-to-speech translation prototype.

**Read [LIMITATIONS.md](LIMITATIONS.md) first:** SignSpeak currently handles 50 isolated ASL signs, one way
(sign to speech), and hasn't yet been tested with Deaf signers.

## Pipeline

```
webcam -> capture -> recognition -> translation -> tts -> audio
          (landmarks) (glosses)     (sentence)    (speech)
```

| Module | Responsibility |
|---|---|
| `src/capture/` | Webcam frames + MediaPipe landmark extraction |
| `src/recognition/` | Sign classification from landmark sequences |
| `src/translation/` | Gloss sequence -> natural sentence |
| `src/tts/` | Text-to-speech with prosody |
| `src/pipeline/` | Glues stages together, manages latency |

Other directories: `data/raw/`, `data/processed/`, `models/` (weights, not committed), `demo/`, `tests/`.

## Setup

Requires Python 3.10 (MediaPipe does not support the newest Python releases).

```powershell
py -3.10 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The TTS engine is still a placeholder in `requirements.txt`. Pick one before working on `src/tts/`.

**MediaPipe note:** MediaPipe 1.x drops the legacy `mp.solutions` API. Use the Tasks API
(`mediapipe.tasks.python.vision`, e.g. `HandLandmarker`), which needs a `.task` model file in `models/`.

Download the Holistic model (not committed; `models/` is gitignored):

```powershell
curl.exe -L -o models\holistic_landmarker.task https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task
```

## Capture

`src/capture/` turns each webcam frame into one 356-value feature vector:

```python
from src.capture import HolisticTracker, extract_features, open_webcam

cap = open_webcam(0)
with HolisticTracker() as tracker:
    ok, frame = cap.read()
    features = extract_features(tracker.process(frame))  # np.float32, shape (356,)
```

The vector has named sections (`LAYOUT["face.nmm"]`, `LAYOUT["right_hand.shape"]`, ...) for both hands,
the upper-body pose, and the face regions used for non-manual markers (eyebrows, lips, head roll/pitch/yaw).
Hand shape is normalized by hand size, and positions are measured relative to the shoulders and scaled by
shoulder width, so signs don't depend on where you stand or how far you are from the camera. A part that
isn't tracked is zero-filled with its `present` flag set to 0. The full layout is documented in
[src/capture/features.py](src/capture/features.py).

Check tracking quality live (landmarks drawn over the webcam feed, plus the vector's live readings):

```powershell
python -m demo.live_capture          # q quits, m toggles mirror view
```

### Sign segmentation

Signs span many frames. `SignSegmenter` keeps a rolling window of the last N feature vectors (default 30)
and cuts the stream into one segment per sign. For now it uses a motion heuristic: a sign starts when hand
motion rises above a threshold and ends after a short pause or when the hands drop. This is a placeholder
for a learned segmentation model.

```python
from src.capture import segment_signs

for segment in segment_signs(stream_of_features_and_timestamps):
    segment.features   # (num_frames, 356)
```

Try it live. The console prints `sign segment captured: <n> frames` each time a sign closes:

```powershell
python -m demo.segment_signs                            # q quits
python -m demo.segment_signs --start 0.6 --stop 0.3     # tune thresholds using the on-screen motion bar
```

## Training data

`data/prepare_dataset.py` turns public sign datasets into the same 356-value per-frame features the live
pipeline produces. It runs the videos through the same Stage 1 tracker, resampled to the live frame rate,
and saves train/val/test tensors to `data/processed/<dataset>/`.

| Dataset | Language | Vocabulary used | License |
|---|---|---|---|
| [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) | **ASL** | 50 signs, ~31 webcam clips each, signer-independent splits | Microsoft Research License: non-commercial research only, no redistribution of the data or anything derived from it |
| [INCLUDE](https://zenodo.org/records/4010759) | **ISL** | 40 signs, ~14-21 clips each | CC BY 4.0 |

> **Limitation: ISL coverage is partial.** INCLUDE is the only substantial open ISL dataset found. Its 263
> words cover greetings, pronouns, adjectives, and time words, but it has no signs for *yes*, *no*,
> *question*, *help*, *please*, *sorry*, question words (*what/why/how/...*), or numbers. The Q&A core
> vocabulary exists only for ASL. ISL and ASL are different languages, so the ASL data can't stand in for ISL.
> Serving ISL users properly needs more ISL data, for example recordings you collect yourself.

The vocabularies live in [data/vocabulary.py](data/vocabulary.py). Clips are streamed straight out of the
datasets' zip archives with HTTP range requests, so nothing near the full archive size is downloaded or
stored. Runs are resumable: re-run the same command after an interruption.

```powershell
python -m data.prepare_dataset asl_citizen     # ~1,550 clips, ~0.8 GB streamed, 1-2 hours on CPU
python -m data.prepare_dataset include         # ~730 clips, ~11 GB streamed; Zenodo is slow (~0.5 MB/s): 10+ hours
python -m data.prepare_dataset include --signs hello "thank you" --max-per-sign 2    # quick trial
```

Each `<split>.pt` holds `features` (N, T, 356) zero-padded, `lengths`, `labels`, `label_names`,
`mirrored`, and `clip_ids`. `meta.json` records the source, license, citation, per-label counts, and
feature layout.

Clips where the signer's *left* hand does most of the moving are mirrored (`mirror_features`), so the
dominant hand always occupies the `right_hand` slot. This covers left-handed signers and mirrored
recordings. The live classifier must apply the same rule to each segment (`left_hand_dominant` in
`data/prepare_dataset.py`). Pass `--keep-handedness` to turn this off.

## Recognition

`src/recognition/` classifies a sequence of feature vectors into one vocabulary sign. The model is a small
bidirectional LSTM (337k parameters) that trains in about 5 minutes on a CPU. `SignRecognizer.predict(segment)`
prepares live segments exactly as training clips were prepared: trimmed to where hands appear, mirrored to
right-hand dominant, and center-cropped.

```powershell
python -m src.recognition.train asl_citizen          # best checkpoint -> models/asl_citizen_sign_classifier.pt
python -m src.recognition.evaluate asl_citizen       # accuracy, per-sign accuracy, confusion matrix (.csv/.png)
python -m src.recognition.baseline asl_citizen       # order-free baseline, for diagnosis
```

Results on ASL Citizen, 50 signs, test signers unseen in training:

| Variant | Test top-1 | Top-5 |
|---|---|---|
| **LSTM, 50 signs** | **74.1%** | 93.6% |
| LSTM, 25 / 10 signs | 81.4% / 86.2% | 94.9% / 99.2% |
| LSTM on 25% / 50% of the training clips | 41.0% / 57.6% | 69.9% / 83.8% |
| Bigger LSTM (2.8M params) | 69.0% | 92.3% |
| Without the right-hand rule | 68.0% | 89.7% |
| Order-free linear baseline | 68.2% | n/a |

**Diagnosis: data is the bottleneck.** Accuracy climbs steeply with more training clips and shows no sign of
flattening, while a model 8× larger does worse. Vocabulary size costs about 6 points per doubling. Most errors
are between signs that look similar (thank you → good, three → two, nine → six).

## Translation

`src/translation/` turns recognized gloss into a sentence: `translate_gloss(["hello", "name", "what"])` →
"Hello! What's your name?".

- **`ClaudeTranslator`** (the default when `ANTHROPIC_API_KEY` is set) prompts a pretrained Claude model with
  the few-shot examples in [examples.py](src/translation/examples.py). Set the model with
  `SIGNSPEAK_TRANSLATION_MODEL` (default `claude-opus-5` at low effort).
- **`RuleBasedTranslator`** ([rules.py](src/translation/rules.py)) is an offline role-based grammar. It is
  used without a key, and whenever the API fails (after an auth or billing error, for the rest of the
  session). It never drops a signed word: sequences it can't parse come back as signed ("Nine language bad six.").
- Any object with `translate(gloss) -> str`, such as a future trained seq2seq model, can be installed with
  `set_default_translator()`.

## Speech

`src/tts/` speaks the sentence with prosody driven by the signer: signing pace sets the speaking rate, and
signs made with an intense facial expression are emphasized.

```python
from src.tts import synthesize
audio = synthesize("I think this idea is really important.", pace=1.35, emphasis_regions=[(5, 7, 1.0)])
audio.save("out.wav")   # or audio.play()
```

- `pace`: 1.0 is normal speed and 1.3 is 30% faster, clamped to 0.6–1.6. `pace_from_durations()`
  derives it from recent sign durations.
- `emphasis_regions`: word spans `(start, end, strength)` by word index. `expression_intensity()` scores a
  sign's facial expression, and `emphasis_for_sentence()` finds the words that sign turned into after translation.

It uses the speech engines built into Windows (OneCore voices, with SAPI5 as fallback): offline, no key. They
were picked by measuring which SSML they actually honor. Rate and per-word relative pitch/volume work on every
engine; `<emphasis>` and named volumes don't, so emphasis is written as explicit pitch, volume and slowdown
(see [prosody.py](src/tts/prosody.py)).

```powershell
python -m demo.tts_demo            # writes demo/tts_samples/*.wav at two settings plus a neutral reference
```

## Live pipeline

`src/pipeline/` wires every stage together: webcam → landmarks → segmentation → classification → gloss
buffer → translation → TTS → speaker. Per-frame work runs on the capture thread; translation, TTS and
playback run on a speech thread so the camera never stalls.

```powershell
python -m src.pipeline.live                         # webcam + preview window + audio
python -m src.pipeline.live --duration 90 --no-window
python -m demo.build_replay_video                   # real test-split signs stitched into sentences
python -m src.pipeline.live --source data/raw/replay/utterances.mp4 --no-window   # replayed in real time
```

Every stage boundary is timestamped on one clock. Each utterance's latency, from **gesture completes** (the
last hand motion of its last sign) to **audio starts** (speech audible at the speaker, after the engine's
silence padding is trimmed), is logged and written to `logs/pipeline_<time>.jsonl`.

Measured on this laptop (i7-7660U, CPU only). Translation used the offline rules, because the API key was invalid:

| | Live webcam (3 utterances) | Replay (6 utterances) |
|---|---|---|
| **End-to-end median** | **1,021 ms** (988–1,340) | **686 ms** (582–947) |
| Under 1.5 s | 3 of 3 | 6 of 6 |
| detect (hand stops → sign closed) | 719 ms (70%) | 516 ms (75%) |
| classify | 125 ms | 16 ms |
| buffer / queue | 0 / 0 ms | 70 / 0 ms |
| translate (offline rules) | 0 ms | 0 ms |
| tts | 47 ms | 16 ms |
| audio start | 90 ms | 98 ms |

**The bottleneck is sign-end detection, driven by tracking speed.** Holistic takes 80–90 ms per frame on this
CPU (10–13 fps), with or without the preview window. The segmenter's closing pause and motion smoothing are
counted in frames, so a slower loop means a later "sign is over" decision. A Claude API translation call
would add a network round trip that hasn't been measured yet.

### Latency optimizations (measured with `python -m src.pipeline.benchmark`)

Each row adds one change. Values are medians over 4 real-time replays per configuration, with frames at
webcam size. "Processing" is end-to-end minus the sentence-boundary waits (`buffer`/`queue`), which depend on
where sentences break rather than on pipeline speed.

| Change | detect | classify | processing | end-to-end |
|---|---|---|---|---|
| Baseline (Stage 7) | 537 ms | 7.5 ms | 673 ms | 760 ms |
| + half-size preview window | 539 ms | 6.0 ms | 680 ms | 730 ms |
| + PyTorch on 1 thread | 581 ms | 5.4 ms | 702 ms | 1,873 ms* |
| + faster sign-end detection (120 ms pause, smoothing 0.8) | **396 ms** | 4.8 ms | **530 ms** | 750 ms |
| + tracking at 640×360 | 390 ms | 4.4 ms | **502 ms** | 715 ms |

\*5 of 11 sentences in that row ended by the 700 ms pause rule instead of hands-down. That's run-to-run
variation, not the thread change.

- **Faster sign-end detection** is the one real win: −140 ms on detect, with no extra split signs on the
  replay (27–30 of 32 signs recognized in every row).
- **Half-size preview** saves ~4 ms/frame of drawing. The window doesn't slow tracking.
- **Tracking at 640×360** saves 10 ms/frame on replay frames, but nothing measurable on live webcam frames
  (~90 ms either way).
- **Not applied:** int8 quantization makes the recognizer 3–7× slower (17 vs 4.4 ms) and costs 1.8 accuracy
  points. TTS renders in 16–47 ms, so streaming has nothing to save.

The pipeline defaults are these tuned settings. Stage 7 settings: `--pause-ms 200 --smoothing 0.6
--preview-scale 1 --track-width 0`.

## Demo UI

A browser dashboard for live demos: the webcam feed with the landmark overlay, recognized signs as they
arrive, the translated sentence, and a latency counter that runs from the end of the sign until the voice
starts, with a per-stage breakdown. Speech plays on the laptop's speakers.

```powershell
python -m demo.app                                                   # webcam; opens http://127.0.0.1:8000
python -m demo.app --source data/raw/replay/utterances.mp4 --loop    # backup: replayed real signing
python -m demo.app --translator rules                                # no network use at all
```

The page loads nothing from the internet, so it works offline. The server (`demo/app.py`, Flask) runs the
pipeline on a background thread. It streams the video as MJPEG and pushes signs, sentences and timings over
Server-Sent Events.

Demo-day tips:

- **Keep the replay backup ready.** It needs no signer, lighting or camera, and runs the same pipeline.
- **Lower your hands at the end of each sentence.** That ends it immediately. Keeping them up means
  waiting for the 700 ms pause rule, the usual cause of a red (over 1.5 s) latency.
- **Sign from the 50-word ASL vocabulary** (listed at the foot of the page), in good light, with both hands
  and your face in frame.
- **Close other heavy apps.** The browser page shares the CPU with tracking: in software-rendered Edge it cut
  the capture loop from ~18–20 to ~14 fps.
- **Unreliable network?** Use `--translator rules`. With the Claude API, a failed call gives up after 3 s and
  speaks the offline translation.

Run the tests:

```powershell
python -m unittest discover -s tests
```
