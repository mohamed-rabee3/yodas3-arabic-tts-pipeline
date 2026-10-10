"""Arabic transcript utilities: normalisation, CER, and caption-guided number -> digit conversion.

Numbers: the tashkeel ASR spells numbers as (diacritised, often dialectal) words. We convert a
number span to Western digits when (a) the independent YouTube caption has digits at the aligned
position and the spoken words parse to the same value, or (b) the span is unambiguous on its own
(multi-word, or value >= 11). A conflict between caption digits and spoken value is reported so
the chunk can be dropped.
"""
import re
from difflib import SequenceMatcher

DIAC = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
EASTERN = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
AMAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ة": "ه", "ى": "ي", "ؤ": "و", "ئ": "ي"})
KEEP = re.compile(r"[^ء-غف-يa-z0-9 ]+")


def strip_tashkeel(t):
    return DIAC.sub("", t or "")


def norm_token(t):
    t = strip_tashkeel(t).translate(EASTERN).translate(AMAP).lower()
    return KEEP.sub("", t)


def norm_text(t):
    """Normalised comparison form: Arabic + Latin letters + digits; no diacritics/punctuation."""
    t = strip_tashkeel(t or "").translate(EASTERN).translate(AMAP).lower()
    t = KEEP.sub(" ", t)
    return " ".join(t.split())


def cer(hyp, ref):
    from rapidfuzz.distance import Levenshtein
    hyp, ref = hyp.replace(" ", ""), ref.replace(" ", "")
    return Levenshtein.distance(hyp, ref) / len(ref) if ref else float("nan")


# ------------------------------------------------------------------ number lexicon (normalised forms)
def _forms(d):
    out = {}
    for v, words in d.items():
        for w in words.split():
            out[norm_token(w)] = v
    return out


UNITS = _forms({
    1: "واحد واحده وحده",
    2: "اثنين اثنان اتنين اثنتين اثنتان تنين",
    3: "ثلاث ثلاثه تلات تلاته تلت",
    4: "اربع اربعه",
    5: "خمس خمسه",
    6: "سته سته ستت",
    7: "سبع سبعه",
    8: "ثمان ثماني ثمانيه ثمنيه تمن تمنيه تمانيه تماني",
    9: "تسع تسعه",
    10: "عشره عشر",
})
TEENS = _forms({
    11: "حداشر احداشر حدعشر احدعشر", 12: "اتناشر اثناشر اطناشر", 13: "تلتاشر ثلطاشر تلطاشر ثلتاشر",
    14: "اربعتاشر اربعطاشر", 15: "خمستاشر خمسطاشر خماستاشر خمصتاشر", 16: "ستاشر سطاشر",
    17: "سبعتاشر سبعطاشر", 18: "تمنتاشر تمنطاشر ثمنتاشر", 19: "تسعتاشر تسعطاشر",
})
TENS = _forms({
    20: "عشرين عشرون", 30: "ثلاثين ثلاثون تلاتين", 40: "اربعين اربعون", 50: "خمسين خمسون",
    60: "ستين ستون", 70: "سبعين سبعون", 80: "ثمانين ثمانون تمانين", 90: "تسعين تسعون",
})
HUNDREDS = _forms({
    100: "ميه مائه مئه مايه ميت", 200: "ميتين مئتين مائتين مئتان مائتان",
    300: "تلتميه ثلاثمائه ثلاثميه ثلثميه تلاتميه ثلاثمئه", 400: "ربعميه اربعمائه اربعميه اربعمئه",
    500: "خمسميه خمسمائه خمسمئه", 600: "ستميه ستمائه ستمئه", 700: "سبعميه سبعمائه سبعمئه",
    800: "تمنميه ثمانمائه ثمنميه تمانميه ثمانمئه", 900: "تسعميه تسعمائه تسعمئه",
})
THOUS = _forms({1000: "الف", 2000: "الفين الفان", -1000: "الاف الوف"})  # -1000: multiplier form
HUNDRED_W = {norm_token(w) for w in ("ميه", "مائه", "مئه", "مايه")}
TEN_W = {norm_token(w) for w in ("عشر", "عشره")}
THOUSAND_MULT_W = {norm_token(w) for w in ("الاف", "الوف", "آلاف")}
SCALE_WORDS = {"مليون", "ملايين", "مليار", "مليارات"}  # kept as words, like the captions ("650 مليون")
STANDALONE_AMBIG = {norm_token(w) for w in ("واحد", "واحده", "وحده", "ست", "سبع", "خمس", "عشر", "تسع", "ثمان", "تلت", "الف")}


def _strip_wa(tok):
    """'و' conjunction prefix -> (had_wa, bare)."""
    if len(tok) > 2 and tok.startswith("و"):
        bare = tok[1:]
        if _value_word(bare) is not None:
            return True, bare
    return False, tok


def _value_word(t):
    for table in (UNITS, TEENS, TENS, HUNDREDS, THOUS):
        if t in table:
            return table[t]
    return None


def parse_number(tokens):
    """Greedy parse of normalised tokens from position 0. Returns (value, n_tokens) or (None, 0).
    Handles: units/teens/tens, 'X و Y' compounds, hundreds, MSA teens ('خمسه عشر'),
    unit+'ميه'/'مائه' split hundreds, thousands ('ثلاثه الاف', 'الفين'), '<n> الف'."""
    total, cur, i, used = 0, 0, 0, 0
    last_kind = None
    while i < len(tokens):
        wa, t = _strip_wa(tokens[i]) if i > 0 else (False, tokens[i])
        if i > 0 and not wa and last_kind not in ("unit_for_teen", "unit_for_scale"):
            # without 'و', only a following scale/teen word may continue the number
            v = _value_word(t)
            if not (v in (1000, -1000) or t in TEN_W or t in HUNDRED_W):
                break
        v = _value_word(t)
        if v is None:
            break
        if v in (1000, -1000):
            if v == -1000:
                cur = (cur or 1) * 1000
            elif cur:
                cur *= 1000
            else:
                cur = 1000
            total += cur
            cur = 0
            last_kind = "thousand"
        elif v == 2000:
            total += 2000
            last_kind = "thousand"
        elif v >= 100:
            if t in HUNDRED_W and last_kind == "unit_for_scale" and cur and cur < 10:
                cur = cur * 100
            else:
                cur += v
            last_kind = "hundred"
        elif t in TEN_W and last_kind == "unit_for_teen" and cur and cur < 10:
            cur += 10
            last_kind = "teen"
        elif v >= 20:
            cur += v
            last_kind = "ten"
        else:
            if v < 10 and cur and last_kind in ("ten",):
                break  # tens are followed by units only in 'unit و ten' order in Arabic
            cur += v
            last_kind = "unit_for_teen" if v < 10 else "teen"
            if v < 10:
                # a unit may also be followed by ميه/الاف without 'و'
                nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
                if nxt in HUNDRED_W or nxt in THOUSAND_MULT_W:
                    last_kind = "unit_for_scale"
        i += 1
        used = i
    val = total + cur
    return (val, used) if used else (None, 0)


def caption_guided_digits(asr_text, caption):
    """Return (text_with_digits, n_converted, n_conflicts).
    asr_text: diacritised ASR output. caption: YouTube caption (may contain digits)."""
    toks = (asr_text or "").translate(EASTERN).split()
    ntok = [norm_token(t) for t in toks]
    cap = norm_text(caption or "").split()
    sm = SequenceMatcher(a=ntok, b=cap, autojunk=False)
    out, i, conv, conflicts = [], 0, 0, 0
    # map ASR index -> caption digit tokens in the corresponding replaced region
    cap_digits_for = {}
    for tag, a0, a1, b0, b1 in sm.get_opcodes():
        if tag in ("replace", "insert") or tag == "equal":
            digs = [c for c in cap[b0:b1] if c.isdigit()]
            if digs and tag != "equal":
                cap_digits_for[a0] = (a1, digs)
    while i < len(toks):
        v, n = parse_number(ntok[i:])
        if v is not None and n:
            span_end = i + n
            guided = None
            for a0, (a1, digs) in cap_digits_for.items():
                if a0 <= i < max(a1, a0 + 1):
                    guided = digs
                    break
            unambiguous = n >= 2 or v >= 11 or ntok[i] not in STANDALONE_AMBIG
            if guided is not None:
                if str(v) in guided:
                    out.append(str(v))
                    conv += 1
                    i = span_end
                    continue
                conflicts += 1
            elif unambiguous and (n >= 2 or v >= 11):
                out.append(str(v))
                conv += 1
                i = span_end
                continue
        out.append(toks[i])
        i += 1
    return " ".join(out), conv, conflicts


if __name__ == "__main__":
    tests = [
        ("مُسْتَشْفَى السَّلَامْ يِصْرَفْ لَهُ سِتِّمِيَّةْ وَخَمْسِينْ مِلْيُونْ دِينَارْ", "السلام يصرف له 650 مليون دينار"),
        ("فِي شَهْرَ عَشَرَةْ مُوسِمْ الْأَهْلِي", "في شهر 10 موسم الاهلي"),
        ("إِحْنَا فِي أَلْفِينْ وِكَامْ دِلْوَاتِي خَمْسَةْ وَعِشْرِينْ", "احنا في 2000 وكام دلوقتي 25"),
        ("عَشْرَةْ خَمَاسْتَاشَرْ سَنَةْ", "10 15 سنه"),
        ("وَاحِدْ قَالْ لِي", "واحد قال لي"),
        ("ثَلَاثَةُ آلَافٍ وَخَمْسُمِائَةِ رَجُلٍ", "ثلاثه الاف وخمسمائه رجل"),
        ("خَمْسَةَ عَشَرَ يَوْمًا", "خمسه عشر يوما"),
        ("سَبْعَةْ أَيَّامْ", "7 ايام"),
        ("ثَلَاثُمِائَةٍ وَخَمْسَةٌ وَسِتُّونَ يَوْمًا", "ثلاثمائه وخمسه وستون يوما"),
        ("أَلْفٌ وَتِسْعُمِائَةٍ وَأَرْبَعَةٌ وَثَمَانُونَ", "1984"),
        ("ثَلَاث مِيَّةْ جُنَيْهْ", "300 جنيه"),
        ("سِتّ الْبِيتْ", "ست البيت"),
        ("خَمْسَةْ أَيَّامْ", "سبعه ايام"),
    ]
    for a, c in tests:
        print(caption_guided_digits(a, c), "|", c)


# ------------------------------------------------------------------ exact word agreement
import unicodedata


def word_key(t):
    """Comparison key for one word: no tashkeel, no punctuation, unified alef/hamza/ta-marbuta/ya forms."""
    return norm_token(unicodedata.normalize("NFKC", t or ""))


L_PRON = {"لك", "له", "لها", "لهم", "لكم", "لي", "لنا", "لكو", "لهن"}


def _ortho(ks):
    """Merge spelling-only variants of the same dialect word (applied to both sides):
    Egyptian negation 'ما حدش' / 'ما كانش' == 'محدش' / 'مكانش';  'يقول لك' == 'يقولك'."""
    out, i = [], 0
    while i < len(ks):
        k = ks[i]
        if k == "ما" and i + 1 < len(ks) and ks[i + 1].endswith("ش") and len(ks[i + 1]) >= 3:
            out.append("م" + ks[i + 1])
            i += 2
            continue
        if k.startswith("ما") and k.endswith("ش") and len(k) >= 5:
            k = "م" + k[2:]
        if k in L_PRON and out and out[-1].endswith("ل"):
            out[-1] = out[-1] + k[1:]
            i += 1
            continue
        out.append(k)
        i += 1
    return out


def words_of(text):
    return _ortho([k for k in (word_key(w) for w in (text or "").split()) if k])


def exact_word_match(asr_text, cap):
    """True if the ASR words equal the caption words (spacing between words ignored).
    `cap` is caption_window() output; caption words that start within the jitter margin of a
    chunk edge may be counted on either side of that edge, nothing else is tolerated."""
    hyp = "".join(words_of(asr_text))
    core = words_of(" ".join(cap["core"]))
    prev = words_of(" ".join(cap["prev"]))
    nxt = words_of(" ".join(cap["next"]))
    if not hyp or not core:
        return False
    for i in range(len(prev) + 1):              # borrow last i words from before the chunk
        for j in range(1 if i else cap["head_n"] + 1):   # or drop first j core words
            for k in range(len(nxt) + 1):        # borrow first k words after the chunk
                for l in range(1 if k else cap["tail_n"] + 1):  # or drop last l core words
                    mid = core[j:len(core) - l]
                    if not mid:
                        continue
                    cand = (prev[len(prev) - i:] if i else []) + mid + nxt[:k]
                    if "".join(cand) == hyp:
                        return True
    return False
