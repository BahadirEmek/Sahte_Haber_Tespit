"""Streamlit interface for the AI-supported fake news detection system."""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from predict import MODEL_PATH, predict_news


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "screenshots" / "confusion_matrix.png"
MIN_WORD_COUNT = 5

CUSTOM_CSS = """
<style>
/* ── Reset / Chrome ───────────────────────────────── */
#MainMenu, footer { visibility: hidden; }

/* ── App background ───────────────────────────────── */
.stApp { background-color: #f4f6fb; }

/* ── Header card ──────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #0f3460 0%, #1a2744 100%);
    border-radius: 14px;
    padding: 1.6rem 2rem;
    margin-bottom: 1.25rem;
}
.app-header h1 {
    color: #ffffff;
    font-size: 1.55rem;
    font-weight: 700;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.3px;
}
.app-header p {
    color: rgba(255,255,255,0.72);
    font-size: 0.85rem;
    margin: 0;
}

/* ── Notice banners ───────────────────────────────── */
.notice-warn {
    background: #fffbeb;
    border-left: 4px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 0.65rem 1rem;
    font-size: 0.83rem;
    color: #78350f;
    margin-bottom: 0.6rem;
}
.notice-info {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 0.65rem 1rem;
    font-size: 0.83rem;
    color: #1e3a8a;
    margin-bottom: 0.6rem;
}

/* ── Result cards ─────────────────────────────────── */
.result-fake {
    background: #fff0f0;
    border: 1px solid #fca5a5;
    border-left: 5px solid #ef4444;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}
.result-fake .result-label {
    color: #b91c1c;
    font-size: 1.1rem;
    font-weight: 700;
}
.result-real {
    background: #f0fdf4;
    border: 1px solid #86efac;
    border-left: 5px solid #22c55e;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}
.result-real .result-label {
    color: #15803d;
    font-size: 1.1rem;
    font-weight: 700;
}
.result-sub {
    color: #6b7280;
    font-size: 0.8rem;
    margin-top: 0.15rem;
}

/* ── Signal items ─────────────────────────────────── */
.signal-item {
    background: #fefce8;
    border-left: 3px solid #ca8a04;
    border-radius: 0 6px 6px 0;
    padding: 0.4rem 0.75rem;
    margin: 0.3rem 0;
    font-size: 0.82rem;
    color: #713f12;
}
.signal-ok {
    background: #f0fdf4;
    border-left: 3px solid #16a34a;
    border-radius: 0 6px 6px 0;
    padding: 0.4rem 0.75rem;
    margin: 0.3rem 0;
    font-size: 0.82rem;
    color: #14532d;
}

/* ── Section label ────────────────────────────────── */
.sec-label {
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #9ca3af;
    margin: 0.9rem 0 0.4rem 0;
}

/* ── ELA info box ─────────────────────────────────── */
.ela-info {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.65rem 0.9rem;
    font-size: 0.8rem;
    color: #475569;
    margin-top: 0.4rem;
}

/* ── Primary button ───────────────────────────────── */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #0f3460, #1a2744) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.6rem !important;
    transition: opacity 0.15s !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    opacity: 0.88 !important;
}
</style>
"""

_CAT_FAKE = "Sahte Haber Ornekleri"
_CAT_REAL = "Gercek Haber Ornekleri"

EXAMPLE_NEWS = {
    _CAT_FAKE: [
        "Bilim insanlari, her sabah ac karnina uc bardak limonlu su icmenin diyabeti yedi gun icinde tamamen iyilestirdigini acikladi. Ismini vermek istemeyen uzmanlara gore ilac sirketleri bu bilgiyi yillardir halktan sakliyor.",
        "Sosyal medyada yayilan iddialara gore, yeni gelistirilen bir telefon uygulamasi insanlarin dusuncelerini okuyabiliyor. Uzmanlar, uygulamanin ozellikle gece saatlerinde aktif hale geldigini ve kisisel bilgileri gizlice topladigi one suruyor.",
        "NASA'nin gizli bir raporuna gore onumüzdeki ay Dunya'nin yaninda ikinci bir ay gorunecek. Raporda bu olayin deniz seviyelerini yukseltecegi ve dunya genelinde hava sicakliklarini aniden degistirecegi iddia edildi.",
        "Bir arastirmaya gore, ders kitabini yastIginin altina koyarak uyuyan ogrenciler bilgileri gece boyunca ezberleyebiliyor. Arastiirmacilar, beynin uyku sirasinda kagit uzerindeki bilgileri dogrudan algiladigini savundu.",
        "Internette yayilan bir habere gore, bankalar onumüzdeki cuma gunu tum bireysel borclari sifIrlayacak. Iddiaya gore vatandaslarin yalnizca bir internet formu doldurmasi, kredi ve kredi karti borclarinin tamamen silinmesi icin yeterli olacak.",
    ],
    _CAT_REAL: [
        "Turkiye Cumhuriyeti 29 Ekim 1923 tarihinde ilan edilmistir. Cumhuriyetin ilanindan sonra Ankara, Turkiye Cumhuriyeti'nin baskenti olarak kabul edilmistir.",
        "Gunes enerjisi, yenilenebilir enerji kaynaklari arasinda yer alir. Gunes panelleri araciligiyla elektrik uretimi yapilabilir ve bu yontem fosil yakit baginimliligini azaltmaya yardimci olur.",
        "Mobil bankacilik uygulamalari, kullanicilarin para transferi, fatura odeme ve hesap hareketlerini takip etme gibi islemleri subeye gitmeden yapabilmesini saglar.",
        "Makine ogrenmesi modelleri, verilerden oruntüler ogrenerek tahmin veya siniflandirma yapabilir. Bu yontem dogal dil isleme, goruntu isleme ve veri analizi gibi alanlarda kullanilmaktadir.",
        "Siber guvenlikte guclu parola kullanimi, iki asamali dogrulama ve guncel yazilim kullanimi temel guvenlik onlemleri arasinda yer alir.",
    ],
}


# ── Session state ─────────────────────────────────────────────────────────────

def _init_state() -> None:
    st.session_state.setdefault("news_text", "")
    st.session_state.setdefault("uploaded_file_name", None)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _example_label(index: int | None, category: str) -> str:
    if index is None:
        return "Ornek secin"
    prefix = "Sahte haber" if category.startswith("Sahte") else "Gercek haber"
    return f"{prefix} {index + 1}"


def _load_example(key: str, category: str) -> None:
    idx = st.session_state.get(key)
    if idx is not None:
        st.session_state.news_text = EXAMPLE_NEWS[category][idx]


def _handle_txt_upload(uploaded_file) -> None:
    if uploaded_file is None:
        return
    if uploaded_file.name != st.session_state.uploaded_file_name:
        st.session_state.news_text = uploaded_file.read().decode("utf-8", errors="ignore")
        st.session_state.uploaded_file_name = uploaded_file.name


def _load_metrics() -> dict | None:
    if not METRICS_PATH.exists():
        return None
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


# ── Tab 1 — Metin Analizi ─────────────────────────────────────────────────────

def render_text_tab() -> None:
    input_col, result_col = st.columns([1.2, 1])

    with input_col:
        st.markdown('<p class="sec-label">Ornek Haberler</p>', unsafe_allow_html=True)
        fake_tab, real_tab = st.tabs(["Sahte Ornekler", "Gercek Ornekler"])

        with fake_tab:
            st.selectbox(
                "Ornek sec",
                options=[None, *range(len(EXAMPLE_NEWS[_CAT_FAKE]))],
                format_func=lambda i: _example_label(i, _CAT_FAKE),
                key="fake_choice",
                on_change=_load_example,
                args=("fake_choice", _CAT_FAKE),
                label_visibility="collapsed",
            )
        with real_tab:
            st.selectbox(
                "Ornek sec",
                options=[None, *range(len(EXAMPLE_NEWS[_CAT_REAL]))],
                format_func=lambda i: _example_label(i, _CAT_REAL),
                key="real_choice",
                on_change=_load_example,
                args=("real_choice", _CAT_REAL),
                label_visibility="collapsed",
            )

        st.markdown('<p class="sec-label">Dosyadan Yukle</p>', unsafe_allow_html=True)
        _handle_txt_upload(st.file_uploader("TXT dosyasi", type=["txt"], label_visibility="collapsed"))

        st.markdown('<p class="sec-label">Haber Metni</p>', unsafe_allow_html=True)
        news_text: str = st.text_area(
            "Metin",
            height=240,
            placeholder="Analiz edilecek haberi buraya yazin veya yapistirin...",
            key="news_text",
            label_visibility="collapsed",
        )

        if not MODEL_PATH.exists():
            st.warning("Model bulunamadi — once `python src/train_model.py` calistirin.")

        analyze_btn = st.button("Haberi Analiz Et", type="primary", use_container_width=True)

    with result_col:
        if analyze_btn:
            if len(news_text.split()) < MIN_WORD_COUNT:
                st.warning("Daha uzun bir metin girin (en az 5 kelime).")
            else:
                with st.spinner("Analiz ediliyor..."):
                    try:
                        result = predict_news(news_text)
                        _render_text_result(result)
                    except FileNotFoundError:
                        st.error("Model bulunamadi. `python src/train_model.py` ile egitim yapiniz.")
                    except ValueError as exc:
                        st.error(str(exc))


def _render_text_result(result: dict) -> None:
    is_fake = result["prediction"] == 0
    confidence = float(result.get("confidence", 0.0))
    card_cls = "result-fake" if is_fake else "result-real"
    label = result["label"]

    st.markdown(
        f'<div class="{card_cls}">'
        f'<div class="result-label">{label}</div>'
        f'<div class="result-sub">Guven skoru: %{confidence:.2f}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )

    st.progress(min(max(confidence / 100, 0.0), 1.0))

    important_terms = result.get("important_terms", [])
    if important_terms:
        st.markdown('<p class="sec-label">Tahmini Etkileyen Kelimeler</p>', unsafe_allow_html=True)
        st.caption(", ".join(important_terms))

    with st.expander("Temizlenmis metin"):
        st.write(result.get("cleaned_text", ""))


# ── Tab 2 — Model Performansi ─────────────────────────────────────────────────

def render_model_tab() -> None:
    metrics = _load_metrics()

    if metrics is None:
        st.warning("Model metrikleri bulunamadi. Once `python src/train_model.py` calistirin.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{float(metrics.get('accuracy', 0)):.4f}")
    col2.metric("Precision", f"{float(metrics.get('precision', 0)):.4f}")
    col3.metric("Recall", f"{float(metrics.get('recall', 0)):.4f}")
    col4.metric("F1-Score", f"{float(metrics.get('f1_score', 0)):.4f}")

    st.markdown('<p class="sec-label">Egitim Ozeti</p>', unsafe_allow_html=True)
    ic1, ic2, ic3 = st.columns(3)
    ic1.metric("Toplam Ornek", metrics.get("total_samples", "—"))
    ic2.metric("Egitim", metrics.get("train_samples", "—"))
    ic3.metric("Test", metrics.get("test_samples", "—"))

    if CONFUSION_MATRIX_PATH.exists():
        st.markdown('<p class="sec-label">Confusion Matrix</p>', unsafe_allow_html=True)
        st.image(str(CONFUSION_MATRIX_PATH))

    cr = metrics.get("classification_report")
    if cr and isinstance(cr, dict):
        with st.expander("Siniflandirma Raporu"):
            for cls_name in ("Fake News", "Real News"):
                if cls_name in cr:
                    row = cr[cls_name]
                    st.write(
                        f"**{cls_name}** — "
                        f"Precision: {row.get('precision', 0):.4f} | "
                        f"Recall: {row.get('recall', 0):.4f} | "
                        f"F1: {row.get('f1-score', 0):.4f}"
                    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Sahte Haber Tespit",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    _init_state()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown(
        '<div class="app-header">'
        "<h1>Yapay Zeka Destekli Sahte Haber Tespit Sistemi</h1>"
        "<p>Metin analizi ile haberlerin gercekligini degerlendirin — "
        "TF-IDF + Logistic Regression</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="notice-warn">'
        "Bu sistem haberin dogruluğunu internette kontrol etmez; egitim verisinden ogrendigi "
        "dilsel oruntulere gore makine ogrenmesi tahmini uretir."
        "</div>"
        '<div class="notice-info">'
        "Model Ingilizce veri setiyle egitildiginden Turkce metinlerde sonuclar demo amacli degerlendirilmelidir."
        "</div>",
        unsafe_allow_html=True,
    )

    tab_text, tab_model = st.tabs(["Metin Analizi", "Model Performansi"])

    with tab_text:
        render_text_tab()

    with tab_model:
        render_model_tab()


if __name__ == "__main__":
    main()
