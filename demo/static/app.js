"use strict";

const $ = (id) => document.getElementById(id);
const TARGET_MS = 1500;
const STAGES = [
  ["detect", "Sign end", "#60a5fa"],
  ["classify", "Recognition", "#a78bfa"],
  ["buffer", "Sentence end", "#fbbf24"],
  ["queue", "Waiting", "#fb923c"],
  ["translate", "Translation", "#2dd4bf"],
  ["tts", "Voice", "#f472b6"],
  ["audio_start", "Audio out", "#34d399"],
];
const STATUS_TEXT = {
  starting: "Warming up",
  listening: "Listening",
  signing: "Signing",
  speaking: "Speaking",
  error: "Error",
  stopped: "Stopped",
  reconnecting: "Reconnecting",
};

let status = "starting";
let chipsSpoken = false; // the chips on screen belong to a sentence already spoken
let ticker = null; // running latency counter
let toastTimer = null;

// ---------------------------------------------------------------- helpers

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function grade(ms) {
  return ms < 1000 ? "good" : ms < TARGET_MS ? "ok" : "slow";
}

function formatMs(ms) {
  return Math.round(ms).toLocaleString("en-US");
}

function toast(message) {
  const box = $("toast");
  box.textContent = message;
  box.classList.add("on");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => box.classList.remove("on"), 4000);
}

// ---------------------------------------------------------------- status

function setStatus(next, message) {
  status = next;
  const pill = $("status");
  pill.dataset.state = next;
  $("status-text").textContent = STATUS_TEXT[next] || next;

  const overlay = $("video-overlay");
  const blocking = next === "starting" || next === "error" || next === "stopped";
  overlay.classList.toggle("hidden", !blocking);
  overlay.classList.toggle("error", next === "error" || next === "stopped");
  if (blocking) $("overlay-text").textContent = message || (next === "starting" ? "Warming up the models…" : "Stopped.");
  $("speaking").classList.toggle("on", next === "speaking");
}

function setInfo(info) {
  if (info.mode) $("mode").textContent = info.mode;
  const parts = [];
  if (info.translator) parts.push(`Translation: ${info.translator}`);
  if (info.voice) parts.push(`Voice: ${info.voice}`);
  if (info.labels && info.labels.length) parts.push(`ASL, ${info.labels.length}-sign vocabulary`);
  $("engine-info").textContent = parts.join("  ·  ");
  if (info.labels && info.labels.length) {
    const vocab = $("vocab");
    vocab.replaceChildren(...info.labels.map((label) => el("span", "", label)));
    $("vocab-count").textContent = `(${info.labels.length})`;
  }
}

// ---------------------------------------------------------------- signs

function addChip(sign) {
  const box = $("glosses");
  if (chipsSpoken || box.querySelector(".placeholder")) {
    box.replaceChildren();
    chipsSpoken = false;
  }
  const chip = el("span", sign.kept ? "chip" : "chip unsure");
  chip.title = sign.kept ? `${Math.round(sign.confidence * 100)}% confident` : "Too unsure to say";
  chip.append(el("span", "word", sign.label));
  const bar = el("span", "bar");
  const fill = el("i");
  fill.style.width = `${Math.max(6, Math.round(sign.confidence * 100))}%`;
  bar.append(fill);
  chip.append(bar);
  box.append(chip);
}

function showChips(chips) {
  const box = $("glosses");
  if (!chips.length) {
    box.replaceChildren(el("p", "placeholder", "Sign in front of the camera."));
    return;
  }
  box.replaceChildren();
  chips.forEach(addChip);
}

function markChipsSpoken() {
  document.querySelectorAll("#glosses .chip:not(.unsure)").forEach((chip) => chip.classList.add("done"));
  document.querySelectorAll("#glosses .chip.unsure").forEach((chip) => chip.remove());
  chipsSpoken = true;
}

// ---------------------------------------------------------------- latency

function showLatency(ms, live) {
  const value = $("latency-value");
  value.textContent = ms === null ? "—" : formatMs(ms);
  value.className = "value " + (ms === null ? "" : live ? "live" : grade(ms));
  const note = $("latency-note");
  note.textContent = ms === null ? "" : live ? "timing…" : "last sentence";
  note.className = live ? "note live" : "note";
  const fill = $("target-fill");
  const share = ms === null ? 0 : Math.min(ms / (TARGET_MS / 0.75), 1); // the 1.5 s mark sits at 75%
  fill.style.width = `${share * 100}%`;
  fill.style.background = ms === null ? "" : live ? "var(--accent)" : `var(--${grade(ms)})`;
}

function startTicker(elapsedMs) {
  // Counts up from when the gesture ended until the sentence is heard.
  stopTicker();
  const origin = performance.now() - elapsedMs;
  const tick = () => {
    const ms = performance.now() - origin;
    if (ms > 8000) {
      // nothing was spoken for it: put the last measured value back
      stopTicker();
      const last = $("history").querySelector(".badge");
      return showLatency(last ? Number(last.textContent.replace(/[^0-9]/g, "")) : null, false);
    }
    showLatency(ms, true);
    ticker = requestAnimationFrame(tick);
  };
  ticker = requestAnimationFrame(tick);
}

function stopTicker() {
  if (ticker !== null) cancelAnimationFrame(ticker);
  ticker = null;
}

function showBreakdown(stages) {
  const total = STAGES.reduce((sum, [key]) => sum + Math.max(0, stages[key] || 0), 0) || 1;
  $("breakdown").replaceChildren(
    ...STAGES.map(([key, , color]) => {
      const segment = el("span");
      segment.style.background = color;
      segment.style.flex = `${Math.max(0, stages[key] || 0) / total}`;
      return segment;
    }),
  );
  $("breakdown-legend").replaceChildren(
    ...STAGES.filter(([key]) => key !== "queue" || (stages.queue || 0) >= 1).map(([key, name, color]) => {
      const item = el("li");
      const swatch = el("i");
      swatch.style.setProperty("--c", color);
      item.append(swatch, document.createTextNode(name), el("b", "", `${formatMs(Math.max(0, stages[key] || 0))} ms`));
      return item;
    }),
  );
}

function showSession(summary) {
  if (!summary.count) {
    $("session").textContent = "No sentences yet";
    return;
  }
  const share = Math.round((100 * summary.under_target) / summary.count);
  $("session").textContent =
    `Median ${formatMs(summary.median_ms)} ms over ${summary.count} sentence${summary.count === 1 ? "" : "s"}` +
    `  ·  ${share}% under 1.5 s`;
}

// ---------------------------------------------------------------- sentences

function historyItem(entry) {
  const item = el("li");
  item.append(el("span", "time", entry.time));
  const said = el("span", "said", entry.sentence);
  said.append(el("span", "signed", entry.glosses.join(" · ")));
  item.append(said);
  item.append(el("span", `badge ${grade(entry.end_to_end)}`, `${formatMs(entry.end_to_end)} ms`));
  return item;
}

function showHistory(entries) {
  const list = $("history");
  if (!entries.length) {
    list.replaceChildren(el("li", "placeholder", "Spoken sentences will appear here."));
    return;
  }
  list.replaceChildren(...entries.map(historyItem));
}

function showSentence(text) {
  const sentence = $("sentence");
  sentence.textContent = text || "—";
  sentence.classList.remove("fresh");
  void sentence.offsetWidth; // restart the fade-in
  sentence.classList.add("fresh");
}

// ---------------------------------------------------------------- events

function handle(event) {
  switch (event.type) {
    case "snapshot": {
      setInfo(event);
      setStatus(event.status, event.message);
      $("fps").textContent = event.fps ? `${event.fps.toFixed(1)} fps` : "— fps";
      showChips(event.current || []);
      showHistory(event.history || []);
      const last = (event.history || [])[0];
      if (last) {
        showSentence(last.sentence);
        showLatency(last.end_to_end, false);
        showBreakdown(last.stages);
      }
      const latencies = event.latencies || [];
      const sorted = [...latencies].sort((a, b) => a - b);
      showSession({
        count: latencies.length,
        median_ms: sorted.length ? sorted[Math.floor((sorted.length - 1) / 2)] : 0,
        under_target: latencies.filter((v) => v < TARGET_MS).length,
      });
      break;
    }
    case "status":
      setInfo(event);
      setStatus(event.status, event.message);
      break;
    case "signing":
      $("signing-badge").classList.toggle("on", event.value);
      if (event.value && status === "listening") setStatus("signing");
      else if (!event.value && status === "signing") setStatus("listening");
      break;
    case "stats":
      $("fps").textContent = `${event.fps.toFixed(1)} fps`;
      break;
    case "sign":
      addChip(event);
      if (event.kept) startTicker(event.elapsed_ms);
      break;
    case "utterance":
      startTicker(event.elapsed_ms);
      break;
    case "spoken":
      stopTicker();
      markChipsSpoken();
      showSentence(event.sentence);
      showLatency(event.end_to_end, false);
      showBreakdown(event.stages);
      showSession(event);
      $("history").querySelector(".placeholder")?.remove();
      $("history").prepend(historyItem(event));
      while ($("history").children.length > 8) $("history").lastChild.remove();
      setStatus("speaking");
      break;
    case "speech_done":
      if (status === "speaking") setStatus("listening");
      break;
    case "notice":
      toast(event.message);
      break;
    case "cleared":
      showHistory([]);
      showChips([]);
      showSentence("");
      showLatency(null, false);
      $("breakdown").replaceChildren();
      $("breakdown-legend").replaceChildren();
      showSession({ count: 0 });
      break;
  }
}

function connect() {
  const source = new EventSource("/events");
  source.onmessage = (message) => handle(JSON.parse(message.data));
  source.onerror = () => {
    if (status !== "error" && status !== "stopped") setStatus("reconnecting");
  };
}

// The video is an endless MJPEG stream. Starting it only after the page has
// loaded keeps the browser from showing the page as "still loading" forever.
function startVideo() {
  const video = $("video");
  video.src = `/video?t=${Date.now()}`;
  // Fit the frame to the source's shape (webcam 16:9, replay 4:3) once the first image arrives.
  const fit = setInterval(() => {
    if (video.naturalWidth > 0) {
      video.parentElement.style.aspectRatio = `${video.naturalWidth} / ${video.naturalHeight}`;
      clearInterval(fit);
    }
  }, 200);
}
// If the stream drops (e.g. the server restarts), reconnect it.
$("video").addEventListener("error", () => setTimeout(startVideo, 1000));
$("clear").addEventListener("click", () => fetch("/api/clear", { method: "POST" }));
if (document.readyState === "complete") startVideo();
else window.addEventListener("load", startVideo);
connect();
