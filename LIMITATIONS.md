# Limitations

SignSpeak is a hackathon prototype. It recognizes a small set of isolated American Sign Language (ASL) signs
from a webcam and speaks an English sentence built from them. This page lists what it can't do yet and where
it breaks, with the numbers we measured. Sign languages are complete languages with their own grammar.
This prototype handles a narrow slice of one of them, and it is not a substitute for a human interpreter.

## At a glance

| What we measured | Result |
|---|---|
| Vocabulary | **50 ASL signs** (ASL has many thousands) |
| Recognition, held-out signers (dataset test split) | **74.1%** top-1, 93.6% top-5 |
| Full demo flow, 10 scripted sentences (26 signs) | **58–65%** of signs recognized and spoken correctly; **3 of 10** sentences fully right |
| Latency, end of sign → voice starts (full demo flow) | median **697–810 ms**; all 27 sentences under 1.5 s |
| Pre-demo audit, 5 single signs through the demo app, run twice | **2–3 of 5** spoken, **1–2 correct**; which signs fail changes between runs; latency 654–904 ms |
| Frame rate on the real webcam (this laptop) | **6–10 fps**, and it swings with whatever else the machine is doing (replays ran at 7–15) |
| Crashes in the final 10-trial runs | **0** (three bugs found and fixed during testing, below) |

## 1. Small vocabulary, not a full sign language

- **50 ASL signs**, chosen for a presentation or Q&A setting: *yes, no, help, please, thank you,
  what/why/how/who/where/when, understand, again, …* plus the numbers one to nine.
- **Gaps within that setting:** *question, repeat, explain, agree, five* and *ten* are missing because the
  dataset doesn't have them.
- **Isolated signs only.** Each sign is classified on its own. There is no fingerspelling (so no names or new
  words) and none of ASL's grammar that lives in space, movement or the face (classifiers, directional verbs,
  role shift).
- **One variant per sign.** Where the dataset has several versions of a sign (e.g. WHAT1, WHAT2), we trained
  on the one with the most examples. Anyone signing another version will likely be misread.

## 2. ASL only; ISL is not supported

- **The working model is ASL**, trained on
  [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/) (Microsoft Research).
- **Indian Sign Language (ISL) is not supported.** The only substantial open ISL dataset we found,
  [INCLUDE](https://zenodo.org/records/4010759), has no signs for *yes, no, question, help, please, sorry*,
  question words, or numbers, which is the core of a Q&A vocabulary. We built its preprocessing pipeline and
  processed a 6-clip trial, but trained no ISL model. ASL and ISL are different languages, so the ASL model
  says nothing about ISL.
- **Limited variety of signers.** ASL Citizen has 52 signers (35 in our training split), recorded on their
  own webcams, and about 15 training clips per sign. It won't cover regional or community variation (for
  example Black ASL) or signers who differ from that group in age or signing style.
- **Handedness is normalized.** Clips where the left hand leads are mirrored so the dominant hand is always
  in the same slot; without this, test accuracy fell from 74.1% to 68.0%. It works, but it is a workaround
  for small data, not a model that understands handedness.
- **License:** ASL Citizen is for non-commercial research only, and neither the data nor anything derived
  from it (including our trained model) may be redistributed. INCLUDE is CC BY 4.0.

## 3. One direction only: sign to speech

SignSpeak turns signing into speech. It does not turn speech into sign (no avatar or sign generation), so a
hearing person's reply reaches a Deaf user only through other means (captions, interpreter). It supports one
side of a conversation, not both.

## 4. Sign segmentation is a basic heuristic

A sign is taken to start when the hands move fast enough and to end after a short pause.

- **Signs without a pause between them merge.** In the full-flow trials, *why problem* came out as one
  segment. Natural fluent signing often has no pause between signs.
- **Signs are missed at low frame rates.** Offline on the same 26 test signs: 25 of 26 found at 15 fps and
  26 at 10 fps, but only 21 at 7.5 fps. In the live demo flow it found 21–23 of 26 at 9.5–13.8 fps.
- **Short signs are detected unreliably, and it is a coin flip, not a threshold.** *where* and *four*, each
  about 0.7 s long, show hands in only 3–8 frames. Fed evenly, the segmenter finds all five audit signs at
  8, 10, 12 and 15 fps. Live, on the same clips, *four* was missed at 11.9 fps in one run and found in
  another, and *where* was missed once and found once. Which frames the loop happens to sample decides it,
  so a sign held for under a second may or may not register.
- **Sentences split when the hands drop mid-sentence.** Lowering the hands ends the sentence. Only 4–6 of
  10 test sentences were spoken whole. That is partly an artifact of the test videos (clips of different
  people stitched together), but real signers do rest their hands too.
- **Keeping the hands up delays speech.** If the hands stay up after the last sign, the system waits
  700 ms to decide the sentence is over, which is the usual cause of latencies over 1 s.
- **Not calibrated per signer.** Thresholds were tuned on replays of dataset clips, not on a range of real
  signers.

## 5. Recognition accuracy depends on data

- **74.1% top-1 on signers the model never saw**, falling to 58–65% in the full live flow, where
  segmentation errors and frame drops add up.
- **Errors are between similar-looking signs:** *thank you → good* (same starting handshape at the chin),
  *three → two*, *nine → six*, *help → discuss*.
- **Data is the bottleneck.** Training on 25% / 50% / 100% of the data gave 41% / 58% / 74%, still rising
  steeply. A model 8× larger did *worse* (69%). Fewer signs would help a little (86% on 10 signs), but more
  clips per sign would help most.
- **Unsure signs are left out.** Predictions below 25% confidence are dropped rather than spoken, so a
  sentence can silently lose a word.

## 6. Translation (gloss → English)

- **The language-model path is untested.** It is built (Claude, few-shot prompted), but our API key was
  rejected for the whole project, so **every sentence in every test came from the offline rule-based
  grammar**. The key is refused by the server itself, not misread by our code: a request with no key gets a
  different error, and our key gets exactly what an invented key of the same shape gets.
- **The API would cost at least 300 ms per sentence.** An empty round trip to the API, rejected at
  authentication with nothing generated, measured 311 ms on our network (plus a 456 ms TLS handshake on the
  first call). Generation time is on top of that and we have never measured it. Against a measured pipeline
  median of about 750 ms, the API path would leave little of the 1.5 s budget.
- **The offline rules are telegraphic outside their templates:** *"I discuss."*, *"Language."*,
  *"Two examples are important."*, *"Who help need?"*, *"I finish time."*, *"It's deaf."* (where *"I'm
  deaf"* is the likely meaning). When a sequence doesn't fit a template, it is spoken exactly as signed,
  never invented.
- **Some sentences are fixed phrases.** The rules include 9 whole-sentence phrases (e.g. UNDERSTAND NO →
  *"I don't understand."*). Two of our 10 test sentences are among them, so the trial translations look
  better than the grammar does on new combinations.
- **The offline rules are now the default**, so the demo needs no key and no network. Choosing the API with
  `--translator claude` costs about 4 s of startup (10 s on a loaded machine) before the first frame, which
  is why it is opt-in.
- **The face isn't used for grammar.** In ASL, raised eyebrows can turn a statement into a yes/no question.
  We track the face and use expression intensity for vocal emphasis, but translation ignores it.

## 7. Speech output

- **Windows only:** it uses the voices built into Windows (OneCore / SAPI5) and a Windows audio path.
- **Emphasis and pace are unvalidated.** Speaking rate follows signing speed, and facial expression adds
  emphasis, but the neutral face and intensity scales are rough estimates. No signer or listener has judged
  whether the prosody feels right.

## 8. Speed and hardware

- **CPU only** (tested on a 2-core i7-7660U laptop). Body, hand and face tracking takes 45–90 ms per frame on
  replayed clips and 96–106 ms on the real webcam with a person in view.
- **The real webcam runs slower than our replay tests.** The camera delivers 27 fps, but the loop managed
  8.3 fps at the old 1280×720 capture and 6.3 fps with the page open, versus 9.5–15 fps on replays. Our
  accuracy numbers come from replays, so live accuracy is likely lower. The camera now opens at 640×480,
  which removes a decode and a 10 ms resize per frame and gained 1.9 fps measured back to back.
- **Tracking cost is the model, not wasted work.** Per frame at best: 78 ms for the bundled holistic model,
  against 106 ms for running pose, hands and face as three separate models, so the bundle is already the
  cheaper path. Input resolution barely changes it (187 ms at 320×240 vs 234 ms at 1280×720 under load,
  because the models resize internally), no GPU delegate is available in this MediaPipe build, and it
  already uses about 1.5 of this laptop's 2 physical cores. Dropping face tracking would save roughly a
  quarter of the cost but would remove the non-manual markers that drive emphasis.
- **Other programs decide the frame rate.** With OneDrive syncing and browsers open (CPU at 100%), per-frame
  tracking rose from 78 ms to 130–220 ms. Across five identical single-sign runs the loop varied between
  7.0 and 13.9 fps purely from background load. Close other programs before demoing; this project folder
  is inside OneDrive, so its sync is itself a competitor.
- **The demo page competes with tracking.** With the dashboard open in a software-rendered browser, the loop
  fell from 13.8 to 9.5 fps, and accuracy fell with it (65% → 58% of signs). A normal browser with GPU
  acceleration, or viewing the page from a second device (`python -m demo.app --host 0.0.0.0`), should help,
  but we haven't measured that.
- **Recognition itself tolerates the lower frame rate.** The test set scored 74.1% at 15 fps, 74.3% at 10 fps
  and 74.4% at 7.5 fps. The loss at low frame rates comes from segmentation, not the classifier.
- **Latency excludes the camera.** It is measured from the last frame showing hand motion. The camera's own
  capture delay (typically 1–2 frames) isn't included.
- **A slower translation is not in these numbers.** The latency figures use offline translation (under 1 ms).
  A real language-model call would add a network round trip we couldn't measure.

## 9. How we tested, and what that doesn't show

- **Test data:** end-to-end tests replay real ASL Citizen test-split clips from signers the model never
  trained on, stitched into sentences. These are isolated dictionary-style signs, not natural continuous
  signing.
- **Almost no live signing:** the live webcam runs captured only a handful of sentences. In the pre-demo
  audit, the only webcam-to-speaker sentence came from an incidental hand movement (heard as *learn*,
  spoken as "I learn." in 726 ms), which proves the path works but says nothing about accuracy.
- **No Deaf users yet.** SignSpeak has **not been tested with Deaf signers or with the Deaf community**.
  That is the most important next step, and it should come before any claims of usefulness.
- **Demo Clip mode shows best cases.** The demo app's recorded clips were chosen because they work: high
  accuracy signs, high-confidence clips, each checked to come out right at 4–15 fps. They run through the
  real models, but their success rate says nothing about typical accuracy (74.1% on unseen signers, lower
  live), and they are single isolated signs.
- **Small samples:** 10 scripted sentences, and benchmark runs of 11–15 sentences per configuration.

## 10. Privacy

- **Video stays on the laptop.** It is processed locally and never stored.
- **If the language-model translator is enabled**, the recognized sign words (text only, no video) are sent
  to Anthropic's API.

## Bugs found and fixed while testing

- **The video feed failed if the page opened during warm-up.** The stream crashed with HTTP 500 because it
  tried to send a frame before the camera had produced one. It now waits for the first frame; covered by a
  regression test.
- **Signs were dropped at low frame rates.** The "too short to be a sign" check measured motion one frame
  short, so at 7.5 fps real signs were discarded as twitches (8 of 26 lost). Fixing it raised signs found
  from 18 to 21 at 7.5 fps without inventing extra ones at higher frame rates.
- **Speech pace was wrong on the real webcam.** Sign durations were computed as if the camera ran at the
  15 fps the model was trained on. At the webcam's ~8 fps every sign looked twice as fast, so speech would
  always have played at its fastest setting. Replays ran near 15 fps, which hid it. Durations now come from
  real frame timestamps; covered by a regression test.
