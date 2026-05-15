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

EXAMPLE_NEWS = {
    "Sahte Haber Örnekleri": [
        "Bilim insanları, her sabah aç karnına üç bardak limonlu su içmenin diyabeti yedi gün içinde tamamen iyileştirdiğini açıkladı. İsmini vermek istemeyen uzmanlara göre ilaç şirketleri bu bilgiyi yıllardır halktan saklıyor.",
        "Sosyal medyada yayılan iddialara göre, yeni geliştirilen bir telefon uygulaması insanların düşüncelerini okuyabiliyor. Uzmanlar, uygulamanın özellikle gece saatlerinde aktif hale geldiğini ve kişisel bilgileri gizlice topladığını öne sürüyor.",
        "NASA’nın gizli bir raporuna göre önümüzdeki ay Dünya’nın yanında ikinci bir ay görünecek. Raporda bu olayın deniz seviyelerini yükselteceği ve dünya genelinde hava sıcaklıklarını aniden değiştireceği iddia edildi.",
        "Bir araştırmaya göre, ders kitabını yastığının altına koyarak uyuyan öğrenciler bilgileri gece boyunca ezberleyebiliyor. Araştırmacılar, beynin uyku sırasında kağıt üzerindeki bilgileri doğrudan algıladığını savundu.",
        "İnternette yayılan bir habere göre, bankalar önümüzdeki cuma günü tüm bireysel borçları sıfırlayacak. İddiaya göre vatandaşların yalnızca bir internet formu doldurması, kredi ve kredi kartı borçlarının tamamen silinmesi için yeterli olacak.",
    ],
    "Gerçek Haber Örnekleri": [
        "Türkiye Cumhuriyeti 29 Ekim 1923 tarihinde ilan edilmiştir. Cumhuriyetin ilanından sonra Ankara, Türkiye Cumhuriyeti'nin başkenti olarak kabul edilmiştir.",
        "Güneş enerjisi, yenilenebilir enerji kaynakları arasında yer alır. Güneş panelleri aracılığıyla elektrik üretimi yapılabilir ve bu yöntem fosil yakıtlara olan bağımlılığı azaltmaya yardımcı olur.",
        "Mobil bankacılık uygulamaları, kullanıcıların para transferi, fatura ödeme ve hesap hareketlerini takip etme gibi işlemleri şubeye gitmeden yapabilmesini sağlar.",
        "Makine öğrenmesi modelleri, verilerden örüntüler öğrenerek tahmin veya sınıflandırma yapabilir. Bu yöntem doğal dil işleme, görüntü işleme ve veri analizi gibi alanlarda kullanılmaktadır.",
        "Siber güvenlikte güçlü parola kullanımı, iki aşamalı doğrulama ve güncel yazılım kullanımı temel güvenlik önlemleri arasında yer alır.",
    ],
}


def read_uploaded_text(uploaded_file) -> str:
    """Read a UTF-8 text file uploaded through Streamlit."""
    if uploaded_file is None:
        return ""

    return uploaded_file.read().decode("utf-8", errors="ignore")


def initialize_session_state() -> None:
    """Initialize Streamlit session values used by the app."""
    st.session_state.setdefault("news_text", "")
    st.session_state.setdefault("uploaded_file_name", None)


def render_sidebar() -> None:
    """Render static project information in the sidebar."""
    st.sidebar.header("Proje Bilgisi")
    st.sidebar.write("Kullanılan Model: TF-IDF + Logistic Regression")
    st.sidebar.write("Sınıflar: Fake News / Real News")
    st.sidebar.write("Veri Seti: Fake and Real News Dataset")
    st.sidebar.write("Geliştirme Ortamı: Python + Streamlit + scikit-learn")


def example_label(index: int | None, category: str) -> str:
    """Return a compact label for an example selectbox option."""
    if index is None:
        return "Örnek seçin"

    prefix = "Sahte haber" if category.startswith("Sahte") else "Gerçek haber"
    return f"{prefix} {index + 1}"


def load_selected_example(selectbox_key: str, category: str) -> None:
    """Copy the selected example text into the editable news text area."""
    selected_index = st.session_state.get(selectbox_key)
    if selected_index is not None:
        st.session_state.news_text = EXAMPLE_NEWS[category][selected_index]


def render_example_selector(category: str, selectbox_key: str) -> None:
    """Render one tab of selectable sample news texts."""
    st.selectbox(
        "Örnek haber seç",
        options=[None, *range(len(EXAMPLE_NEWS[category]))],
        format_func=lambda index: example_label(index, category),
        key=selectbox_key,
        on_change=load_selected_example,
        args=(selectbox_key, category),
    )


def handle_uploaded_file(uploaded_file) -> None:
    """Move uploaded text file content into the editable text area."""
    if uploaded_file is None:
        return

    if uploaded_file.name != st.session_state.uploaded_file_name:
        st.session_state.news_text = read_uploaded_text(uploaded_file)
        st.session_state.uploaded_file_name = uploaded_file.name


def render_examples() -> None:
    """Render selectable fake and real sample news tabs."""
    st.subheader("Örnek Haber Metinleri")
    fake_tab, real_tab = st.tabs(["Sahte Haber Örnekleri", "Gerçek Haber Örnekleri"])

    with fake_tab:
        render_example_selector("Sahte Haber Örnekleri", "fake_example_choice")

    with real_tab:
        render_example_selector("Gerçek Haber Örnekleri", "real_example_choice")


def render_prediction_result(result: dict) -> None:
    """Render prediction result, confidence, important terms, and cleaned text."""
    confidence = float(result.get("confidence", 0.0))
    st.subheader("Analiz Sonucu")

    if result["prediction"] == 0:
        st.error(f"Sonuç: {result['label']}")
    else:
        st.success(f"Sonuç: {result['label']}")

    st.metric("Confidence", f"%{confidence:.2f}")
    st.progress(min(max(confidence / 100, 0.0), 1.0))

    important_terms = result.get("important_terms", [])
    if important_terms:
        st.subheader("Tahmini Etkileyen Kelimeler")
        st.write(", ".join(important_terms))

    with st.expander("Temizlenmiş metin"):
        st.write(result.get("cleaned_text", ""))


def load_metrics() -> dict | None:
    """Load saved training metrics when available."""
    if not METRICS_PATH.exists():
        return None

    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def render_model_performance() -> None:
    """Render model performance metrics and confusion matrix if they exist."""
    st.subheader("Model Performansı")
    metrics = load_metrics()

    if metrics is None:
        st.warning("Model performans metrikleri için önce model eğitimi yapılmalıdır.")
        return

    metric_columns = st.columns(4)
    metric_values = [
        ("Accuracy", metrics.get("accuracy", 0)),
        ("Precision", metrics.get("precision", 0)),
        ("Recall", metrics.get("recall", 0)),
        ("F1-score", metrics.get("f1_score", 0)),
    ]

    for column, (label, value) in zip(metric_columns, metric_values):
        column.metric(label, f"{float(value):.4f}")

    if CONFUSION_MATRIX_PATH.exists():
        st.image(str(CONFUSION_MATRIX_PATH), caption="Confusion Matrix")


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(
        page_title="Yapay Zeka Destekli Sahte Haber Tespit Sistemi",
        layout="wide",
    )
    initialize_session_state()
    render_sidebar()

    st.title("Yapay Zeka Destekli Sahte Haber Tespit Sistemi")
    st.write("Sistem haber metnini analiz ederek Fake News / Real News tahmini üretir.")
    st.warning(
        "Bu sistem haberin doğruluğunu internette kontrol etmez; eğitim verisinden öğrendiği dilsel örüntülere göre makine öğrenmesi tahmini üretir."
    )
    st.info(
        "Model İngilizce veri setiyle eğitildiği için Türkçe metinlerde sonuçlar demo amaçlı değerlendirilmelidir."
    )

    input_column, result_column = st.columns([1.2, 1])

    with input_column:
        render_examples()
        uploaded_file = st.file_uploader("Txt dosyası yükle", type=["txt"])
        handle_uploaded_file(uploaded_file)

        news_text = st.text_area(
            "Haber metni",
            height=300,
            placeholder="Analiz edilecek haber metnini buraya yazın...",
            key="news_text",
        )

        if not MODEL_PATH.exists():
            st.warning("Model dosyası bulunamadı. Lütfen önce model eğitimi yapın.")
            st.code("python src/train_model.py", language="bash")

        analyze_clicked = st.button("Haberi Analiz Et", type="primary")

    with result_column:
        if analyze_clicked:
            if len(news_text.split()) < MIN_WORD_COUNT:
                st.warning("Lütfen analiz için daha uzun bir haber metni girin.")
            else:
                try:
                    result = predict_news(news_text)
                    render_prediction_result(result)
                except FileNotFoundError:
                    st.error("Model dosyası bulunamadı. Lütfen önce model eğitimi yapın.")
                    st.code("python src/train_model.py", language="bash")
                except ValueError as exc:
                    st.error(str(exc))

    st.divider()
    render_model_performance()


if __name__ == "__main__":
    main()
