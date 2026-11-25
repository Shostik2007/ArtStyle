import json
import logging
from io import BytesIO
from pathlib import Path

import keras
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "best_model.keras"
LABEL_MAP_PATH = APP_DIR.parent / "splits" / "label_map.json"
STYLE_INFO_PATH = APP_DIR / "style_info.json"
EXAMPLES_DIR = APP_DIR / "examples"
IMAGE_SIZE = 224
RESIZE_SIDE = 256
CONFIDENCE_THRESHOLD = 0.50
LOGGER = logging.getLogger(__name__)


@st.cache_resource
def load_model():
    return keras.models.load_model(MODEL_PATH, compile=False)


def preprocess(image_bytes):
    # Check dimensions before TensorFlow allocates the decoded and resized images.
    with Image.open(BytesIO(image_bytes)) as source:
        width, height = source.size
        if width * height > 40_000_000 or max(width, height) > 64 * min(width, height):
            raise ValueError("Слишком большое изображение или слишком вытянутые пропорции.")

    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    height_width = tf.cast(tf.shape(image)[:2], tf.float32)
    scale = RESIZE_SIDE / tf.reduce_min(height_width)
    size = tf.cast(tf.round(height_width * scale), tf.int32)
    image = tf.image.resize(image, size, antialias=True)
    image = tf.clip_by_value(image, 0.0, 255.0)
    return tf.image.resize_with_crop_or_pad(image, IMAGE_SIZE, IMAGE_SIZE)


def predict(model, batch, explain):
    if not explain:
        return model(batch, training=False).numpy()[0], None

    backbone = model.get_layer("efficientnetb0")
    classifier = model.get_layer("style_head")
    features = backbone(batch, training=False)
    with tf.GradientTape() as tape:
        tape.watch(features)
        logits = classifier(features, training=False)[0]
        class_id = tf.argmax(logits, output_type=tf.int32)
        score = tf.gather(logits, class_id)

    gradients = tape.gradient(score, features)[0]
    weights = tf.reduce_mean(gradients, axis=(0, 1))
    heatmap = tf.nn.relu(tf.reduce_sum(features[0] * weights, axis=-1))
    heatmap = tf.math.divide_no_nan(heatmap, tf.reduce_max(heatmap))
    heatmap = tf.image.resize(heatmap[..., None], (IMAGE_SIZE, IMAGE_SIZE))[..., 0]
    probabilities = tf.nn.softmax(tf.cast(logits, tf.float32))
    return probabilities.numpy(), heatmap.numpy()


def overlay_heatmap(image, heatmap):
    colors = np.array([[44, 19, 86], [195, 45, 92], [247, 151, 55], [255, 241, 177]])
    stops = np.linspace(0.0, 1.0, len(colors))
    tinted = np.stack([np.interp(heatmap, stops, channel) for channel in colors.T], axis=-1)
    alpha = 0.68 * heatmap[..., None]
    return np.rint(image * (1 - alpha) + tinted * alpha).astype(np.uint8)


def main():
    st.set_page_config(page_title="ArtStyle", layout="wide")
    st.title("ArtStyle")
    st.write("Загрузи картину, чтобы определить её стиль среди семи классов WikiArt.")

    if not MODEL_PATH.is_file():
        st.error("Файл модели отсутствует. Помести best_model.keras в папку app.")
        st.stop()

    controls, result = st.columns([1, 1.4], gap="large")
    with controls:
        uploaded = st.file_uploader("Картина", type=["jpg", "jpeg", "png"])
        example = None
        if uploaded is None and (EXAMPLES_DIR / "examples.json").is_file():
            examples = json.loads((EXAMPLES_DIR / "examples.json").read_text(encoding="utf-8"))
            example = st.selectbox("Или выбери пример", list(examples), index=None, placeholder="Выбери картину")
        explain = st.checkbox("Показать Grad-CAM")

    if uploaded is None and example is None:
        return

    with controls:
        if uploaded is not None:
            image_bytes = uploaded.getvalue()
        else:
            artwork = examples[example]
            image_bytes = (EXAMPLES_DIR / artwork["file"]).read_bytes()
            st.caption(f"[{artwork['source_label']}]({artwork['source_url']})")

        try:
            image = preprocess(image_bytes)
        except ValueError as error:
            st.error(str(error))
            return
        except (UnidentifiedImageError, Image.DecompressionBombError, tf.errors.OpError, OSError):
            st.error("Не удалось прочитать картину. Проверь, что файл JPEG или PNG открывается.")
            return

        crop = np.rint(image.numpy()).astype(np.uint8)
        st.image(crop, caption="Фрагмент, который получила модель", width=320)

    with result:
        try:
            with st.spinner("Анализируем картину…"):
                probabilities, heatmap = predict(load_model(), image[None, ...], explain)
        except (OSError, ValueError, tf.errors.OpError):
            LOGGER.exception("Model inference failed")
            st.error("Не удалось запустить модель. Проверь файл best_model.keras и версии зависимостей.")
            return

        labels = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))["idx_to_label"]
        style_info = json.loads(STYLE_INFO_PATH.read_text(encoding="utf-8"))
        top_indices = np.argsort(-probabilities, kind="stable")[:3]
        st.subheader("Вероятные стили")
        for index in top_indices:
            style = style_info[labels[str(index)]]
            probability = float(probabilities[index])
            st.progress(probability, text=f"**{style['name']}** · {probability:.1%}")

        best = top_indices[0]
        if probabilities[best] < CONFIDENCE_THRESHOLD:
            st.warning("Модель не уверена в прогнозе. Вероятность первого стиля ниже 50%.")
        st.write(style_info[labels[str(best)]]["description"])

        if heatmap is not None:
            st.image(overlay_heatmap(crop, heatmap), caption="Grad-CAM для первого стиля", width=320)


if __name__ == "__main__":
    main()
