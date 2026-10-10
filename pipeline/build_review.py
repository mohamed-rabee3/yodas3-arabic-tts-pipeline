"""Build the listening-review page (HTML) from review samples, shard stats and the AITW audit."""
import html, json
import pandas as pd

samples = json.load(open("/workspace/review/samples.json"))
stats = json.load(open("/workspace/test_out/stats/0000.json"))
auc = pd.read_csv("/workspace/aitw/auc_report.csv").set_index("feature")
n_aitw = len(pd.read_parquet("/workspace/aitw/features.parquet"))

REASON_LABEL = {
    "kept": "Kept", "multi_speaker": "Two or more speakers (pyannote)", "multi_speaker_dz": "Second speaker (DiariZen)",
    "caption_speaker_turn": "Speaker change in captions (>>)", "overlap": "Overlapping speech", "music": "Background music (CED)",
    "music_fr": "Background music (FireRedVAD)", "singing": "Singing", "noisy": "Background noise (DNSMOS)",
    "noisy_um": "Background noise (UrgentMOS)", "low_quality": "Low overall quality", "clipping": "Clipping",
    "cut_edge": "Word cut at an edge", "synthetic": "Synthetic voice", "asr_caption_mismatch": "ASR disagrees with caption", "word_mismatch": "Words differ from caption",
    "bandwidth": "Narrow bandwidth (< 10 kHz)", "crowd_babble": "Crowd or babble",
}
SCORE_LABEL = {"dnsmos_ovrl": "DNSMOS overall", "dnsmos_bak": "DNSMOS background", "um_mos_overall": "UrgentMOS",
               "aes_pc": "Mix complexity", "tag_music": "Music (CED)", "fr_music_p95": "Music (FireRed)",
               "dz_2nd_spk_sec": "2nd speaker s", "spk_min_sim": "Speaker sim", "tag_synth": "Synthetic",
               "bw_hz": "Bandwidth Hz", "clip_frac": "Clipped frac", "asr_cer": "ASR vs caption CER"}

h = lambda x: html.escape(str(x)) if x is not None else ""
audio_h = stats["audio_s"] / 3600
kept_h = stats.get("kept_s", 0) / 3600
rej = sorted(((k[7:], v) for k, v in stats.items() if k.startswith("reject_")), key=lambda x: -x[1])[:10]
rej_max = rej[0][1]


def auc_row(check, rows):
    cells = "".join(f'<td class="num{" win" if i == 0 else ""}">{h(name)}<span>{auc.loc[f, col]:.3f}</span></td>'
                    for i, (name, f, col) in enumerate(rows))
    return f"<tr><th>{h(check)}</th>{cells}</tr>"


audit = "".join([
    auc_row("Two or more speakers", [("DiariZen", "dz_2nd_spk_sec", "multi_speaker"), ("WeSpeaker-34", "spk_min_sim", "multi_speaker"),
                                     ("WeSpeaker-293", "spk293_min_sim", "multi_speaker"), ("pyannote 3.0", "seg_n_spk", "multi_speaker")]),
    auc_row("Background music", [("FireRedVAD", "fr_music_p95", "background_music"), ("CED-base", "tag_music", "background_music"),
                                 ("Audiobox PC", "aes_pc", "background_music"), ("DNSMOS BAK", "dnsmos_bak", "background_music")]),
    auc_row("Background noise", [("UrgentMOS", "um_scoreq", "noise"), ("Audiobox PC", "aes_pc", "noise"),
                                 ("DNSMOS BAK", "dnsmos_bak", "noise"), ("UTMOSv2", "utmosv2", "noise")]),
    auc_row("Synthetic voice", [("CED-base", "tag_synth", "synthetic"), ("UrgentMOS", "um_mos_overall", "synthetic"),
                                ("DNSMOS", "dnsmos_ovrl", "synthetic"), ("UTMOSv2", "utmosv2", "synthetic")]),
])

cats = ["kept"] + [c for c in REASON_LABEL if c != "kept" and any(s["cat"] == c for s in samples)]
filters = "".join(f'<button type="button" class="chip-btn" data-cat="{c}" aria-pressed="{str(c == "all").lower()}">'
                  f'{h(REASON_LABEL.get(c, c))} <span class="cnt">{sum(s["cat"] == c for s in samples)}</span></button>'
                  for c in cats)

cards = []
for s in samples:
    kept = s["cat"] == "kept"
    sc = "".join(f'<div><dt>{h(SCORE_LABEL[k])}</dt><dd>{"—" if v is None else (f"{v:,.0f}" if k == "bw_hz" else v)}</dd></div>'
                 for k, v in s["scores"].items() if k in SCORE_LABEL)
    other = [r for r in (s["fail"] or "").split(",") if r and r != s["cat"]]
    also = f'<p class="also">Also failed: {h(", ".join(REASON_LABEL.get(r, r.replace("_", " ")) for r in other))}</p>' if other else ""
    tash = (f'<p class="lbl">Tashkeel transcript</p><p class="ar" lang="ar" dir="rtl">{h(s["tashkeel"])}</p>'
            if s.get("tashkeel") else "")
    cards.append(f'''
<article class="clip {'is-kept' if kept else 'is-rej'}" data-cat="{s['cat']}" data-id="{h(s['id'])}">
  <header class="clip-head">
    <span class="tag {'tag-keep' if kept else 'tag-rej'}">{h(REASON_LABEL.get(s['cat'], s['cat']))}</span>
    <span class="dur">{s['dur']:.1f} s</span>{'<span class="salv">Re-split piece</span>' if s.get('salvaged') else ''}
    <code class="cid">{h(s['id'])}</code>
  </header>
  <audio controls preload="none" src="data:audio/mpeg;base64,{s['mp3']}"></audio>
  <div class="texts">
    <p class="lbl">YouTube caption</p><p class="ar" lang="ar" dir="rtl">{h(s['caption'] or '(no caption)')}</p>
    {tash}
  </div>
  {also}
  <dl class="scores">{sc}</dl>
  <div class="verdict" role="group" aria-label="Your verdict">
    <button type="button" class="v v-good" id="good-{h(s['id'])}" data-v="good">Sounds clean</button>
    <button type="button" class="v v-bad" id="bad-{h(s['id'])}" data-v="bad">Has a problem</button>
    <span class="saved" aria-live="polite"></span>
  </div>
</article>''')

page = f'''<title>Yodas3 Arabic Listening Desk</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Noto+Naskh+Arabic:wght@400;600&display=swap">
<style>
/* Layout: one reading column; summary + audit up top, then a filterable stack of clip sheets. */
:root {{
  --paper: #f4f6f8; --sheet: #ffffff; --ink: #18212b; --muted: #5b6876; --rule: #d9dfe5;
  --accent: #1d5f8c; --keep: #23704a; --keep-bg: #e3f1e9; --rej: #a8392e; --rej-bg: #f7e6e3;
  --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace;
  --arabic: "Noto Naskh Arabic", "Geeza Pro", "Arial", serif;
  color-scheme: light;
}}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --paper: #10161d; --sheet: #17202a; --ink: #e6ebf0; --muted: #98a6b4; --rule: #2a3542;
  --accent: #6fb3e0; --keep: #6fcf97; --keep-bg: #173326; --rej: #f08a7e; --rej-bg: #3a1d1a; color-scheme: dark; }} }}
:root[data-theme="dark"] {{
  --paper: #10161d; --sheet: #17202a; --ink: #e6ebf0; --muted: #98a6b4; --rule: #2a3542;
  --accent: #6fb3e0; --keep: #6fcf97; --keep-bg: #173326; --rej: #f08a7e; --rej-bg: #3a1d1a; color-scheme: dark; }}
* {{ box-sizing: border-box; }}
body {{ background: var(--paper); color: var(--ink); font: 15px/1.55 var(--sans); padding-inline: 16px; }}
.wrap {{ max-width: 980px; margin: 0 auto; padding-block: 32px 64px; display: grid; gap: 28px; }}
h1 {{ font-size: 1.75rem; font-weight: 600; margin: 0; letter-spacing: -0.01em; text-wrap: balance; }}
h2 {{ font-size: 1.05rem; font-weight: 600; margin: 0 0 10px; text-wrap: balance; }}
.lede {{ max-width: 68ch; color: var(--muted); margin: 6px 0 0; }}
.facts {{ display: flex; flex-wrap: wrap; gap: 8px 28px; font-variant-numeric: tabular-nums; }}
.facts div {{ display: grid; }}
.facts b {{ font: 500 1.35rem/1.2 var(--mono); }}
.facts span {{ font-size: .78rem; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }}
.panel {{ background: var(--sheet); border: 1px solid var(--rule); border-radius: 6px; padding: 18px 20px; }}
.split {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
@media (max-width: 760px) {{ .split {{ grid-template-columns: 1fr; }} }}
.bars {{ display: grid; gap: 6px; font-size: .85rem; }}
.bar {{ display: grid; grid-template-columns: 11rem 1fr 3.5rem; gap: 10px; align-items: center; }}
.bar i {{ display: block; height: 8px; background: var(--rej); border-radius: 2px; opacity: .75; }}
.bar em {{ font: 400 .8rem var(--mono); color: var(--muted); font-style: normal; text-align: right; font-variant-numeric: tabular-nums; }}
.tbl {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: .84rem; }}
th, td {{ text-align: left; padding: 7px 8px; border-bottom: 1px solid var(--rule); vertical-align: top; }}
td span {{ display: block; font: 500 .9rem var(--mono); font-variant-numeric: tabular-nums; }}
td.win {{ color: var(--keep); font-weight: 600; }}
.note {{ font-size: .82rem; color: var(--muted); margin: 10px 0 0; max-width: 72ch; }}
.filters {{ display: flex; flex-wrap: wrap; gap: 6px; position: sticky; top: env(safe-area-inset-top, 0px); background: var(--paper); padding-block: 10px; z-index: 2; }}
.chip-btn {{ font: 500 .8rem var(--sans); color: var(--ink); background: var(--sheet); border: 1px solid var(--rule); border-radius: 999px; padding: 5px 11px; cursor: pointer; }}
.chip-btn[aria-pressed="true"] {{ background: var(--accent); border-color: var(--accent); color: var(--sheet); }}
.chip-btn .cnt {{ font-family: var(--mono); opacity: .7; margin-left: 2px; }}
.progress {{ font: 400 .85rem var(--mono); color: var(--muted); }}
.stack {{ display: grid; gap: 14px; }}
.clip {{ background: var(--sheet); border: 1px solid var(--rule); border-radius: 6px; padding: 16px 18px; display: grid; gap: 12px; }}
.clip.is-kept {{ box-shadow: inset 3px 0 0 var(--keep); }}
.clip.is-rej {{ box-shadow: inset 3px 0 0 var(--rej); }}
.clip-head {{ display: flex; flex-wrap: wrap; gap: 8px 12px; align-items: baseline; }}
.tag {{ font-size: .74rem; font-weight: 600; letter-spacing: .04em; text-transform: uppercase; padding: 2px 8px; border-radius: 3px; }}
.tag-keep {{ color: var(--keep); background: var(--keep-bg); }}
.tag-rej {{ color: var(--rej); background: var(--rej-bg); }}
.dur {{ font: 500 .85rem var(--mono); }}
.salv {{ font-size: .74rem; color: var(--accent); border: 1px solid var(--accent); border-radius: 3px; padding: 1px 6px; }}
.cid {{ font: .72rem var(--mono); color: var(--muted); overflow-wrap: anywhere; }}
audio {{ width: 100%; max-width: 100%; }}
.texts {{ display: grid; gap: 2px; }}
.lbl {{ margin: 6px 0 0; font-size: .72rem; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }}
.ar {{ margin: 0; font: 400 1.22rem/1.9 var(--arabic); }}
.also {{ margin: 0; font-size: .82rem; color: var(--rej); }}
.scores {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 6px 14px; margin: 0; }}
.scores div {{ display: flex; justify-content: space-between; gap: 8px; border-bottom: 1px dotted var(--rule); font-size: .8rem; }}
.scores dt {{ color: var(--muted); }}
.scores dd {{ margin: 0; font-family: var(--mono); font-variant-numeric: tabular-nums; }}
.verdict {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
.v {{ font: 500 .85rem var(--sans); padding: 7px 14px; border-radius: 4px; border: 1px solid var(--rule); background: var(--sheet); color: var(--ink); cursor: pointer; }}
.v-good[aria-pressed="true"] {{ background: var(--keep); border-color: var(--keep); color: var(--sheet); }}
.v-bad[aria-pressed="true"] {{ background: var(--rej); border-color: var(--rej); color: var(--sheet); }}
.saved {{ font-size: .78rem; color: var(--muted); }}
button:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
[hidden] {{ display: none !important; }}
</style>
<div class="wrap">
  <header>
    <h1>YODAS v3 Arabic: listening check before the full run</h1>
    <p class="lede">Clips from a 120-video test of shard 0000, cut at pauses and scored by every judge. Listen to the kept clips first:
    each should be one speaker, no music, no background noise, no clipping and no chopped words, and every word of the tashkeel transcript
    should match the YouTube caption. Clips marked “Re-split piece” were cut from a longer chunk whose other part had a word disagreement. Kept clips play the exact dataset file, full length.
    Rejected clips play at most their first 12 seconds.
    Mark each clip; your verdicts are saved and used to tune the thresholds before all 208 shards are processed.</p>
  </header>
  <section class="facts" aria-label="Test run">
    <div><b>{audio_h:.1f} h</b><span>Audio audited</span></div>
    <div><b>{stats['speech_s']/3600:.1f} h</b><span>Speech found</span></div>
    <div><b>{stats['chunk_s']/3600:.1f} h</b><span>Cut into chunks</span></div>
    <div><b>{kept_h:.2f} h</b><span>Kept ({100*kept_h/audio_h:.1f}%)</span></div>
    <div><b>{stats.get('kept_n',0)}</b><span>Clean chunks</span></div>
  </section>
  <div class="split">
    <section class="panel">
      <h2>Biggest reasons for rejection</h2>
      <div class="bars">{''.join(f'<div class="bar"><span>{h(REASON_LABEL.get(k, k.replace("_", " ")))}</span><i style="width:{100*v/rej_max:.0f}%"></i><em>{v/60:.0f} min</em></div>' for k, v in rej)}</div>
      <p class="note">A chunk can fail several rules; each rule is counted. Minutes of chunk audio.</p>
    </section>
    <section class="panel">
      <h2>Model audit on human labels</h2>
      <div class="tbl"><table><thead><tr><th>Check</th><th colspan="4">ROC-AUC per model (best first)</th></tr></thead><tbody>{audit}</tbody></table></div>
      <p class="note">{n_aitw} clips from AITW (human-labelled YODAS clips). 1.0 means the model separates bad from clean perfectly, 0.5 is chance.
      The pipeline uses the winner of each row, plus the runner-up as a second vote.</p>
    </section>
  </div>
  <nav class="filters" aria-label="Filter clips">
    <button type="button" class="chip-btn" data-cat="all" aria-pressed="true">All <span class="cnt">{len(samples)}</span></button>{filters}
    <span class="progress" id="progress"></span>
  </nav>
  <section class="stack" id="stack">{''.join(cards)}</section>
</div>
<script>
(() => {{
  const clips = [...document.querySelectorAll('.clip')];
  const verdicts = {{}};
  const prog = document.getElementById('progress');
  const paint = () => {{
    let n = 0;
    clips.forEach(c => {{
      const v = verdicts[c.dataset.id];
      if (v) n++;
      c.querySelectorAll('.v').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.v === v)));
    }});
    prog.textContent = n + ' of ' + clips.length + ' marked';
  }};
  document.querySelectorAll('.chip-btn').forEach(btn => btn.addEventListener('click', () => {{
    document.querySelectorAll('.chip-btn').forEach(b => b.setAttribute('aria-pressed', String(b === btn)));
    const cat = btn.dataset.cat;
    clips.forEach(c => {{ c.hidden = !(cat === 'all' || c.dataset.cat === cat); }});
  }}));
  // only one clip plays at a time
  document.addEventListener('play', e => {{ document.querySelectorAll('audio').forEach(a => {{ if (a !== e.target) a.pause(); }}); }}, true);
  let db = null;
  try {{ const raw = localStorage.getItem('verdicts'); if (raw) Object.assign(verdicts, JSON.parse(raw)); }} catch (e) {{}}
  paint();
  clips.forEach(c => c.querySelectorAll('.v').forEach(b => b.addEventListener('click', async () => {{
    const id = c.dataset.id, v = b.dataset.v;
    verdicts[id] = v; paint();
    try {{ localStorage.setItem('verdicts', JSON.stringify(verdicts)); }} catch (e) {{}}
    const out = c.querySelector('.saved');
    if (!db) {{ out.textContent = 'Saved in this browser only'; return; }}
    out.textContent = 'Saving…';
    try {{ await db.collection('verdicts').doc(id).set({{ verdict: v, cat: c.dataset.cat, at: Date.now() }}); out.textContent = 'Saved'; }}
    catch (e) {{ out.textContent = 'Not saved (' + (e && e.code || 'error') + '). It is kept in this browser.'; }}
  }})));
  if (window.claude && window.claude.use) {{
    window.claude.use('db').then(async d => {{
      if (!d) return;
      db = d;
      try {{
        const snap = await db.collection('verdicts').get();
        snap.docs.forEach(s => {{ const x = s.data(); if (x && x.verdict) verdicts[s.id] = x.verdict; }});
        paint();
      }} catch (e) {{}}
    }});
  }}
}})();
</script>
'''
open("/workspace/review/review.html", "w").write(page)
print(len(page) / 1e6, "MB")
