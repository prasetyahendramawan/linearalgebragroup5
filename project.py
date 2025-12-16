import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
from io import BytesIO
import base64

# ===================== CONFIG & THEME =====================

st.set_page_config(
    page_title="🧮 Matrix Transformations in Image Processing",
    layout="wide",
)

def set_video_background(video_path: str):
    """Set an mp4 video as full-screen background using HTML/CSS."""
    if not os.path.exists(video_path):
        st.warning(f"Background video not found: {video_path}")
        return

    with open(video_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    video_data_url = f"data:video/mp4;base64,{b64}"

    html = """
        <style>
        .video-bg {{
            position: fixed;
            right: 0;
            bottom: 0;
            min-width: 100%;
            min-height: 100%;
            width: auto;
            height: auto;
            z-index: -1;
            object-fit: cover;
        }}
        .stApp {{
            background: transparent !important;
        }}
        </style>
        <video class="video-bg" autoplay muted loop playsinline>
            <source src="{video_data_url}" type="video/mp4">
        </video>
    """.format(video_data_url=video_data_url)
    st.markdown(html, unsafe_allow_html=True)

set_video_background("assets/background.mp4")

base_css = """
<style>
.block-container {
    max-width: 1200px;
    padding: 2.5rem 2rem 1.2rem 2rem;
}
.stImage > img{
    max-height:420px;
    object-fit:contain;
}
section[data-testid="stExpander"]{
    border-radius:10px;
    padding:8px;
    box-shadow:0 1px 6px rgba(0,0,0,0.04);
    margin-bottom:10px;
    background-color: var(--stLightBlue-50);
}
section[data-testid="stExpander"] .streamlit-expanderHeader{
    font-size:16px;
}
div[data-testid="column"] button {
    padding-top: 8px !important;
    padding-bottom: 8px !important;
    padding-left: 12px !important;
    padding-right: 12px !important;
    font-size: 14px !important;
    width: 100%;
    font-weight: 500 !important;
}
div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 2px solid #4CAF50 !important;
    border-radius: 12px !important;
}
.team-photo-container {
    width: 140px;
    height: 140px;
    border-radius: 50%;
    overflow: hidden;
    margin: 0 auto;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #f0f0f0;
    border: 3px solid #4CAF50;
}
.team-photo-container img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center;
}
</style>
"""
light_css = """
<style>
.stMarkdown, .stMarkdown p, .stMarkdown li {
    color: #ffffff !important;
}
button[kind="secondary"] {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: 2px solid #4CAF50 !important;
    font-weight: 600 !important;
}
button[kind="secondary"]:hover {
    background-color: #e8f5e9 !important;
    color: #000000 !important;
    border-color: #2e7d32 !important;
}
button[kind="primary"] {
    background-color: #ffffff !important;
    color: #000000 !important;
}
.team-photo-container {
    background: #e8f5e9;
    border-color: #4CAF50;
}
</style>
"""

st.markdown(base_css, unsafe_allow_html=True)
st.markdown(light_css, unsafe_allow_html=True)

# ===================== SESSION STATE =====================

if "original_img" not in st.session_state:
    st.session_state["original_img"] = None
if "geo_transform" not in st.session_state:
    st.session_state["geo_transform"] = None
if "image_filter" not in st.session_state:
    st.session_state["image_filter"] = None

# ===================== TRANSLATIONS (EN ONLY) =====================

translations = {
    "en": {
        "title": "🧮 Matrix Transformations in Image Processing",
        "subtitle": "🎯 Try matrix effects and image adjustments",
        "app_goal": "🎯 *App Goal:* Provide a practical explanation of how 2D matrix transformations and image filters work on photos using linear algebra concepts.",
        "features": "- ↩ Transformations: translation, scaling, rotation, shear, and reflection.\n- 🧽 Processing: blur images, sharpen details, detect edges, remove background, convert to grayscale, and adjust brightness–contrast.",
        "concept_1_title": "### 🌀 2D Matrix Transformations",
        "concept_1_text1": "🌀 Flat images can be seen as a collection of points \\((x, y)\\) whose positions can be changed by linear operations such as translation, scaling, rotation, shear, and reflection, represented by 2×2 or 3×3 matrices (homogeneous coordinates).",
        "concept_1_text2": "🔄 When these matrices are multiplied by the point coordinates, the position shifts: scaling changes size, rotation rotates the image around the center, shear makes the shape oblique, and reflection flips the image over a specific line.",
        "concept_2_title": "### 📊 Image Adjustment with Convolution",
        "concept_2_text1": "📊 Image adjustment using small convolution kernels (matrices) that are shifted across the image; at each position, the new pixel value is calculated from a weighted combination of its neighbors.",
        "concept_2_text2": "🔍 Kernels with more even values produce blur or smoothing effects, while kernels with a strong center and negative values around it can sharpen and highlight edges.",
        "concept_3_title": "### 🎲 Why Interactive?",
        "concept_3_text1": "🎛 Users can adjust parameters like rotation angle, scaling factor, shear strength, or kernel choices and see the results on the image immediately, making matrix formulas feel more concrete.",
        "concept_3_text2": "💡 This way, numbers in the matrix can be directly linked to visual changes, making ideas of linear transformations and convolution easier to understand intuitively.",
        "quick_concepts": "#### 📝 Quick Main Ideas",
        "quick_concepts_text": "- ↩ 2D Transformations: move points in the plane (translation, scaling, rotation, shear, reflection).\n- 📊 Convolution: small kernel shifted over the image to calculate each new pixel from its surroundings.",
        "upload_title": "### 📷 Upload Image",
        "upload_label": "Drop image here (PNG/JPG/JPEG) 📂",
        "upload_success": "✅ Image loaded successfully.",
        "upload_preview": "📷 Original Image Preview",
        "upload_info": "⬆ Please upload an image first before using processing features.",
        "tools_title": "### 🔧 Image Processing Tools",
        "tools_subtitle": "🎛 Select one of the sections below to set transformations or filters.",
        "geo_title": "#### 🔄 Geometric Transformations",
        "geo_desc": "🔄 Geometric transformations change the position, size, and orientation of pixels using matrix-based linear operations.",
        "btn_translation": "↔ Translation",
        "btn_scaling": "📏 Scaling",
        "btn_rotation": "🔄 Rotation",
        "btn_shearing": "📐 Shear",
        "btn_reflection": "🪞 Reflection",
        "geo_info": "🔔 Upload an image first to try geometric transformations.",
        "trans_settings": "↔ Translation Settings**",
        "trans_dx": "↔ dx (shift left–right)",
        "trans_dy": "↕ dy (shift up–down)",
        "btn_apply": "✅ Apply",
        "trans_result": "📷 Translation Result",
        "scale_settings": "📏 Scaling Settings**",
        "scale_x": "📏 X-axis scaling factor",
        "scale_y": "📏 Y-axis scaling factor",
        "scale_result": "📷 Scaling Result",
        "rot_settings": "🔄 Rotation Settings**",
        "rot_angle": "🔄 Rotation angle (degrees)",
        "rot_result": "📷 Rotation Result",
        "shear_settings": "📐 Shear Settings**",
        "shear_x": "📐 X shear factor",
        "shear_y": "📐 Y shear factor",
        "shear_result": "📷 Shear Result",
        "refl_settings": "🪞 Reflection Settings**",
        "refl_axis": "🪞 Reflection axis",
        "refl_result": "📷 Reflection Result",
        "hist_title": "#### 📈 Color Histogram",
        "hist_desc": "📈 Histogram shows the distribution of pixel intensities (dark to light) for each color channel and helps evaluate exposure and contrast.",
        "btn_histogram": "Show Histogram 📈",
        "hist_warning": "⚠ Upload an image first to display histogram.",
        "filter_title": "#### 🔧 Filters and Adjustments",
        "filter_desc": "🔧 Filters change pixel values based on neighboring pixels (convolution) for blur, sharpening, edge detection, background removal, and brightness–contrast adjustment.",
        "btn_blur": "🔲 Blur",
        "btn_sharpen": "✨ Sharpen",
        "btn_background": "🎯 Background",
        "btn_grayscale": "⚫ Grayscale",
        "btn_edge": "🔍 Edge Detection",
        "btn_brightness": "☀ Brightness–Contrast",
        "filter_info": "🔔 Upload an image first to use filters.",
        "blur_settings": "🔲 Blur Settings**",
        "blur_kernel": "🔲 Kernel size",
        "blur_result": "📷 Blur Result",
        "sharpen_settings": "✨ Sharpening Settings**",
        "sharpen_desc": "✨ Highlights details and edges in the image.",
        "sharpen_result": "📷 Sharpening Result",
        "bg_settings": "🎯 Background Removal Settings**",
        "bg_method": "🎯 Method (example using HSV and simple segmentation)",
        "bg_result": "📷 Background Processing Result",
        "gray_settings": "⚫ Grayscale Settings**",
        "gray_desc": "⚫ Converts color image to grayscale.",
        "gray_result": "📷 Grayscale Result",
        "edge_settings": "🔍 Edge Detection Settings**",
        "edge_method": "🔍 Edge detection method",
        "edge_result": "📷 Edge Image",
        "bright_settings": "☀ Brightness & Contrast Settings**",
        "bright_brightness": "☀ Brightness value",
        "bright_contrast": "🌑 Contrast value",
        "bright_result": "📷 Brightness–Contrast Result",
        "team_title": "### 👥 Team Members",
        "team_subtitle": "👥 Group 3 – Roles and contributions",
        "team_sid": "🆔 SID:",
        "team_role": "👤 Role:",
        "team_contribution": "🤝 Contribution:",
        "upload_method_title": "### 📤 How to Upload Image",
        "upload_method_text": "*Image upload steps:\n1. Click the **\"Drop image here (PNG/JPG/JPEG) 📂\"* button at the top of the page.\n2. Select an image file from your device (PNG, JPG, or JPEG).\n3. Wait until the image finishes loading and appears on the screen.\n4. If successful, a confirmation message and original image preview will be displayed.\n5. After that, you can use transformations on the left column and filters on the right column.",
        "team_group": "👥 Group:",
        "axis_x": "➡ X-Axis",
        "axis_y": "⬆ Y-Axis",
        "axis_diag": "↗ Diagonal",
        "nav_label": "🧭 Page Navigation",
        "nav_expl": "📖 Explanation",
        "nav_proc": "🖼 Upload & Processing",
        "nav_team": "👥 Team",
    },
    "id" : {
        "title": "🔢 Operasi Matriks untuk Pemrosesan Visual",
        "subtitle": "🎯 Coba efek matriks 2D dan penyesuaian citra",
        "app_goal": "🎯 *Tujuan aplikasi:* Memberikan penjelasan praktis tentang cara transformasi matriks dua dimensi dan filter citra bekerja pada foto menggunakan konsep aljabar linear.",
        "features": "- ↩ Transformasi: translasi, skala, rotasi, shear, dan refleksi.\n- 🧽 Pemrosesan: menghaluskan citra, menajamkan detail, deteksi tepi, menghapus latar belakang, konversi ke grayscale, serta mengatur kecerahan–kontras.",
        "concept_1_title": "### 🌀 Transformasi Matriks Dua Dimensi",
        "concept_1_text1": "🌀 Gambar datar dapat dilihat sebagai kumpulan titik \\((x, y)\\) yang posisinya dapat diubah oleh operasi linear seperti translasi, skala, rotasi, shear, dan refleksi, yang direpresentasikan oleh matriks 2×2 atau 3×3 (koordinat homogen).",
        "concept_1_text2": "🔄 Ketika matriks ini dikalikan dengan koordinat titik, posisi titik bergeser: skala mengubah ukuran, rotasi memutar gambar terhadap pusat, shear membuat bentuk menjadi miring, dan refleksi membalik gambar terhadap garis tertentu.",
        "concept_2_title": "### 📊 Penyesuaian Citra dengan Konvolusi",
        "concept_2_text1": "📊 Penyesuaian citra menggunakan kernel kecil (matriks konvolusi) yang digeser di seluruh gambar; pada setiap posisi, nilai piksel baru dihitung dari kombinasi berbobot tetangganya.",
        "concept_2_text2": "🔍 Kernel dengan nilai yang lebih merata menghasilkan efek blur atau smoothing, sementara kernel dengan pusat kuat dan nilai negatif di sekitarnya dapat menajamkan dan menonjolkan tepi.",
        "concept_3_title": "### 🎲 Mengapa Interaktif?",
        "concept_3_text1": "🎛 Pengguna dapat mengatur parameter seperti sudut rotasi, faktor skala, kekuatan shear, atau pilihan kernel dan langsung melihat hasilnya pada gambar, sehingga rumus matriks terasa lebih konkret.",
        "concept_3_text2": "💡 Dengan cara ini, angka di dalam matriks bisa langsung dihubungkan dengan perubahan visual, sehingga ide transformasi linear dan konvolusi lebih mudah dipahami secara intuitif.",
        "quick_concepts": "#### 📝 Ide Utama Singkat",
        "quick_concepts_text": "- ↩ Transformasi 2D: memindahkan titik di bidang (translasi, skala, rotasi, shear, refleksi).\n- 📊 Konvolusi: kernel kecil digeser di atas gambar untuk menghitung setiap piksel baru dari lingkungan sekitarnya.",
        "upload_title": "### 📷 Unggah Gambar",
        "upload_label": "Letakkan gambar di sini (PNG/JPG/JPEG) 📂",
        "upload_success": "✅ Gambar berhasil dimuat.",
        "upload_preview": "📷 Pratinjau Gambar Asli",
        "upload_info": "⬆ Silakan unggah gambar terlebih dahulu sebelum memakai fitur pemrosesan.",
        "tools_title": "### 🔧 Image Processing Tools",
        "tools_subtitle": "🎛 Pilih salah satu bagian di bawah ini untuk mengatur transformasi atau filter.",
        "geo_title": "#### 🔄 Transformasi Geometris",
        "geo_desc": "🔄 Transformasi geometris mengubah posisi, ukuran, dan orientasi piksel menggunakan operasi linear berbasis matriks.",
        "btn_translation": "↔ Translasi",
        "btn_scaling": "📏 Skala",
        "btn_rotation": "🔄 Rotasi",
        "btn_shearing": "📐 Shear",
        "btn_reflection": "🪞 Refleksi",
        "geo_info": "🔔 Unggah gambar terlebih dahulu untuk mencoba transformasi geometris.",
        "trans_settings": "↔ Pengaturan Translasi**",
        "trans_dx": "↔ dx (geser kiri–kanan)",
        "trans_dy": "↕ dy (geser atas–bawah)",
        "btn_apply": "✅ Terapkan",
        "trans_result": "📷 Hasil Translasi",
        "scale_settings": "📏 Pengaturan Skala**",
        "scale_x": "📏 Faktor skala sumbu X",
        "scale_y": "📏 Faktor skala sumbu Y",
        "scale_result": "📷 Hasil Skala",
        "rot_settings": "🔄 Pengaturan Rotasi**",
        "rot_angle": "🔄 Sudut rotasi (derajat)",
        "rot_result": "📷 Hasil Rotasi",
        "shear_settings": "📐 Pengaturan Shear**",
        "shear_x": "📐 Faktor shear X",
        "shear_y": "📐 Faktor shear Y",
        "shear_result": "📷 Hasil Shear",
        "refl_settings": "🪞 Pengaturan Refleksi**",
        "refl_axis": "🪞 Sumbu refleksi",
        "refl_result": "📷 Hasil Refleksi",
        "hist_title": "#### 📈 Histogram Warna",
        "hist_desc": "📈 Histogram menunjukkan sebaran intensitas piksel (gelap ke terang) untuk tiap kanal warna dan membantu menilai eksposur serta kontras.",
        "btn_histogram": "Tampilkan Histogram 📈",
        "hist_warning": "⚠ Unggah gambar terlebih dahulu untuk menampilkan histogram.",
        "filter_title": "#### 🔧 Filter dan Penyesuaian Citra",
        "filter_desc": "🔧 Filter mengubah nilai piksel berdasarkan piksel tetangga (konvolusi) untuk blur, sharpening, deteksi tepi, penghapusan latar belakang, dan pengaturan kecerahan–kontras.",
        "btn_blur": "🔲 Blur",
        "btn_sharpen": "✨ Tajamkan",
        "btn_background": "🎯 Background",
        "btn_grayscale": "⚫ Grayscale",
        "btn_edge": "🔍 Deteksi Tepi",
        "btn_brightness": "☀ Kecerahan–Kontras",
        "filter_info": "🔔 Unggah gambar terlebih dahulu untuk menggunakan filter.",
        "blur_settings": "🔲 Pengaturan Blur**",
        "blur_kernel": "🔲 Ukuran kernel",
        "blur_result": "📷 Hasil Blur",
        "sharpen_settings": "✨ Pengaturan Penajaman**",
        "sharpen_desc": "✨ Menonjolkan detail dan tepi pada gambar.",
        "sharpen_result": "📷 Hasil Penajaman",
        "bg_settings": "🎯 Pengaturan Penghapusan Latar Belakang**",
        "bg_method": "🎯 Metode (contoh menggunakan HSV dan segmentasi sederhana)",
        "bg_result": "📷 Hasil Pemrosesan Background",
        "gray_settings": "⚫ Pengaturan Grayscale**",
        "gray_desc": "⚫ Mengubah gambar berwarna menjadi skala abu-abu.",
        "gray_result": "📷 Hasil Grayscale",
        "edge_settings": "🔍 Pengaturan Deteksi Tepi**",
        "edge_method": "🔍 Metode deteksi tepi",
        "edge_result": "📷 Gambar Tepi",
        "bright_settings": "☀ Pengaturan Kecerahan & Kontras**",
        "bright_brightness": "☀ Nilai kecerahan",
        "bright_contrast": "🌑 Nilai kontras",
        "bright_result": "📷 Hasil Kecerahan–Kontras",
        "team_title": "### 👥 Anggota Kelompok",
        "team_subtitle": "👥 Kelompok 3 – Peran dan kontribusi",
        "team_sid": "🆔 NIM:",
        "team_role": "👤 Peran:",
        "team_contribution": "🤝 Kontribusi:",
        "upload_method_title": "### 📤 Cara Mengunggah Gambar",
        "upload_method_text": "*Langkah mengunggah gambar:\n1. Klik tombol **\"Letakkan gambar di sini (PNG/JPG/JPEG) 📂\"* di bagian atas halaman.\n2. Pilih file gambar dari perangkat (PNG, JPG, atau JPEG).\n3. Tunggu sampai gambar selesai dimuat dan muncul di layar.\n4. Jika berhasil, pesan konfirmasi dan pratinjau gambar asli akan ditampilkan.\n5. Setelah itu, kamu dapat menggunakan transformasi di kolom kiri dan filter di kolom kanan.",
        "team_group": "👥 Kelompok:",
        "axis_x": "➡ Sumbu-X",
        "axis_y": "⬆ Sumbu-Y",
        "axis_diag": "↗ Diagonal",
        "nav_label": "🧭 Navigasi halaman",
        "nav_expl": "📖 Penjelasan",
        "nav_proc": "🖼 Unggah & Pemrosesan",
        "nav_team": "👥 Anggota",
    },
}

if "language" not in st.session_state:
    st.session_state["language"] = "en"
lang = "en" if st.session_state["language"] == "en" else "id"
t = translations[lang]
# ===================== HEADER & SIDEBAR NAV =====================

with st.container(border=True):
    header_col1, header_col2 = st.columns([6, 4], vertical_alignment="center")
    with header_col1:
        st.title(t["title"])
    with header_col2:
        st.subheader(t["subtitle"])

page = st.sidebar.radio(
    t["nav_label"],
    [t["nav_expl"], t["nav_proc"], t["nav_team"]],
)
st.sidebar.markdown("*Language / Bahasa:*")
col_lang1, col_lang2 = st.sidebar.columns(2)
with col_lang1:
    if st.button("EN", key="lang_en"):
        st.session_state["language"] = "en"
        st.rerun()
with col_lang2:
    if st.button("ID", key="lang_id"):
        st.session_state["language"] = "id"
        st.rerun()

# ===================== HELPER FUNCTIONS =====================

def load_image(file):
    img = Image.open(file).convert("RGB")
    return np.array(img)

def to_opencv(img_rgb):
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

def to_streamlit(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def apply_affine_transform(img_rgb, M, output_size=None):
    img_bgr = to_opencv(img_rgb)
    h, w = img_bgr.shape[:2]
    if output_size is None:
        output_size = (w, h)
    if M.shape == (3, 3):
        M_affine = M[0:2, :]
    else:
        M_affine = M
    transformed = cv2.warpAffine(
        img_bgr, M_affine, output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return to_streamlit(transformed)

def manual_convolution_gray(img_gray, kernel):
    k_h, k_w = kernel.shape
    pad_h = k_h // 2
    pad_w = k_w // 2
    padded = np.pad(img_gray, ((pad_h, pad_h), (pad_w, pad_w)), mode="reflect")
    h, w = img_gray.shape
    output = np.zeros_like(img_gray, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            region = padded[i:i + k_h, j:j + k_w]
            output[i, j] = np.sum(region * kernel)
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output

def rgb_to_gray(img_rgb):
    img_bgr = to_opencv(img_rgb)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return gray

def adjust_brightness_contrast(img_rgb, brightness=0, contrast=0):
    img_bgr = to_opencv(img_rgb)
    beta = brightness
    alpha = 1 + (contrast / 100.0)
    adjusted = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)
    return to_streamlit(adjusted)

def image_to_bytes(img_rgb, fmt="PNG"):
    if img_rgb is None:
        raise ValueError("image_to_bytes received None image")
    arr = np.array(img_rgb)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if fmt.upper() == "JPEG" and arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    pil_img = Image.fromarray(arr.astype("uint8"))
    buf = BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()

def compute_histogram(img_rgb):
    img_bgr = to_opencv(img_rgb)
    color = ("b", "g", "r")
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, col in enumerate(color):
        hist = cv2.calcHist([img_bgr], [i], None, [256], [0, 256])
        ax.plot(hist, color=col)
        ax.set_xlim([0, 256])
    ax.set_title("Color Histogram")
    ax.set_xlabel("Pixel value")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    return fig

def simple_background_removal_hsv(img_rgb):
    """
    Simple background removal using HSV threshold.
    Assumes background is relatively light and near-neutral.
    """
    img_bgr = to_opencv(img_rgb)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower_bg = np.array([0, 0, 180])      # low saturation, high value
    upper_bg = np.array([180, 60, 255])

    mask_bg = cv2.inRange(hsv, lower_bg, upper_bg)
    mask_fg = cv2.bitwise_not(mask_bg)

    fg_bgr = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_fg)
    fg_rgb = to_streamlit(fg_bgr)
    return fg_rgb

# ===================== PAGE 1: EXPLANATION =====================

if page == t["nav_expl"]:
    expl_col_left, expl_col_right = st.columns(2, vertical_alignment="top")

    with expl_col_left:
        with st.container(border=True):
            st.markdown(t["app_goal"])
            st.markdown(t["features"])

        with st.container(border=True):
            st.markdown(t["quick_concepts"])
            st.markdown(t["quick_concepts_text"])

        with st.container(border=True):
            st.markdown(t["concept_1_title"])
            st.markdown(t["concept_1_text1"])

    with expl_col_right:
        with st.container(border=True):
            st.markdown(t["concept_1_text2"])

        with st.container(border=True):
            st.markdown(t["concept_2_title"])
            st.markdown(t["concept_2_text1"])
            st.markdown(t["concept_2_text2"])

        with st.container(border=True):
            st.markdown(t["concept_3_title"])
            st.markdown(t["concept_3_text1"])
            st.markdown(t["concept_3_text2"])


# ===================== PAGE 2: UPLOAD & PROCESSING =====================

elif page == t["nav_proc"]:

    with st.container(border=True):
        st.markdown(t["upload_title"])
        uploaded_file = st.file_uploader(
            label=t["upload_label"],
            type=["png", "jpg", "jpeg"],
            key="image_uploader_main",
        )
        if uploaded_file is not None:
            original_img = load_image(uploaded_file)
            st.session_state["original_img"] = original_img
            st.success(t["upload_success"])
            st.image(original_img, caption=t["upload_preview"], use_column_width=True)
        else:
            st.info(t["upload_info"])

    original_img = st.session_state["original_img"]

    st.markdown(t["tools_title"])
    st.write(t["tools_subtitle"])

    with st.container(border=True):
        st.markdown(t["upload_method_title"])
        st.markdown(t["upload_method_text"])

    tools_col_left, tools_col_right = st.columns(2, vertical_alignment="top")

    # LEFT: Geometric transforms
    with tools_col_left:
        with st.container(border=True):
            st.markdown(t["geo_title"])
            st.write(t["geo_desc"])
            st.markdown("---")

            row1 = st.columns(3)
            with row1[0]:
                if st.button(t["btn_translation"], key="btn_trans", type="secondary"):
                    st.session_state["geo_transform"] = "translation"
            with row1[1]:
                if st.button(t["btn_scaling"], key="btn_scale", type="secondary"):
                    st.session_state["geo_transform"] = "scaling"
            with row1[2]:
                if st.button(t["btn_rotation"], key="btn_rot", type="secondary"):
                    st.session_state["geo_transform"] = "rotation"

            row2 = st.columns(3)
            with row2[0]:
                if st.button(t["btn_shearing"], key="btn_shear", type="secondary"):
                    st.session_state["geo_transform"] = "shearing"
            with row2[1]:
                if st.button(t["btn_reflection"], key="btn_refl", type="secondary"):
                    st.session_state["geo_transform"] = "reflection"
            with row2[2]:
                pass

        with st.container(border=True):
            if original_img is None:
                st.info(t["geo_info"])
            else:
                mode = st.session_state["geo_transform"]

                if mode == "translation":
                    st.markdown(t["trans_settings"])
                    dx = st.slider(t["trans_dx"], -200, 200, 0, key="dx")
                    dy = st.slider(t["trans_dy"], -200, 200, 0, key="dy")
                    if st.button(f"{t['btn_apply']} ✅", key="apply_trans"):
                        Tm = np.array([[1, 0, dx],
                                       [0, 1, dy],
                                       [0, 0, 1]], dtype=np.float32)
                        out = apply_affine_transform(original_img, Tm)
                        st.image(out, caption=t["trans_result"], use_column_width=True)
                        c_png, c_jpg = st.columns(2)
                        with c_png:
                            st.download_button(
                                "⬇ Download PNG",
                                data=image_to_bytes(out, "PNG"),
                                file_name="translation.png",
                                mime="image/png",
                            )
                        with c_jpg:
                            st.download_button(
                                "⬇ Download JPG",
                                data=image_to_bytes(out, "JPEG"),
                                file_name="translation.jpg",
                                mime="image/jpeg",
                            )

                elif mode == "scaling":
                    st.markdown(t["scale_settings"])
                    sx = st.slider(t["scale_x"], 0.1, 3.0, 1.0, key="sx")
                    sy = st.slider(t["scale_y"], 0.1, 3.0, 1.0, key="sy")
                    if st.button(f"{t['btn_apply']} ✅", key="apply_scale"):
                        h, w = original_img.shape[:2]
                        Sm = np.array([[sx, 0, 0],
                                       [0, sy, 0],
                                       [0, 0, 1]], dtype=np.float32)
                        new_w = int(w * sx)
                        new_h = int(h * sy)
                        out = apply_affine_transform(original_img, Sm, output_size=(new_w, new_h))
                        st.image(out, caption=t["scale_result"], use_column_width=True)
                        c_png, c_jpg = st.columns(2)
                        with c_png:
                            st.download_button(
                                "⬇ Download PNG",
                                data=image_to_bytes(out, "PNG"),
                                file_name="scaling.png",
                                mime="image/png",
                            )
                        with c_jpg:
                            st.download_button(
                                "⬇ Download JPG",
                                data=image_to_bytes(out, "JPEG"),
                                file_name="scaling.jpg",
                                mime="image/jpeg",
                            )

                elif mode == "rotation":
                    st.markdown(t["rot_settings"])
                    angle = st.slider(t["rot_angle"], -180, 180, 0, key="angle")
                    if st.button(f"{t['btn_apply']} ✅", key="apply_rot"):
                        h, w = original_img.shape[:2]
                        cx, cy = w / 2, h / 2
                        theta = np.deg2rad(angle)
                        cos_t = np.cos(theta)
                        sin_t = np.sin(theta)
                        Rm = np.array([[cos_t, -sin_t, 0],
                                       [sin_t,  cos_t, 0],
                                       [0,      0,     1]], dtype=np.float32)
                        T1 = np.array([[1, 0, -cx],
                                       [0, 1, -cy],
                                       [0, 0, 1]], dtype=np.float32)
                        T2 = np.array([[1, 0, cx],
                                       [0, 1, cy],
                                       [0, 0, 1]], dtype=np.float32)
                        M = T2 @ Rm @ T1
                        out = apply_affine_transform(original_img, M)
                        st.image(out, caption=t["rot_result"], use_column_width=True)
                        c_png, c_jpg = st.columns(2)
                        with c_png:
                            st.download_button(
                                "⬇ Download PNG",
                                data=image_to_bytes(out, "PNG"),
                                file_name="rotation.png",
                                mime="image/png",
                            )
                        with c_jpg:
                            st.download_button(
                                "⬇ Download JPG",
                                data=image_to_bytes(out, "JPEG"),
                                file_name="rotation.jpg",
                                mime="image/jpeg",
                            )

                elif mode == "shearing":
                    st.markdown(t["shear_settings"])
                    shx = st.slider(t["shear_x"], -1.0, 1.0, 0.0, key="shx")
                    shy = st.slider(t["shear_y"], -1.0, 1.0, 0.0, key="shy")
                    if st.button(f"{t['btn_apply']} ✅", key="apply_shear"):
                        Sm = np.array([[1,   shx, 0],
                                       [shy, 1,   0],
                                       [0,   0,   1]], dtype=np.float32)
                        out = apply_affine_transform(original_img, Sm)
                        st.image(out, caption=t["shear_result"], use_column_width=True)
                        c_png, c_jpg = st.columns(2)
                        with c_png:
                            st.download_button(
                                "⬇ Download PNG",
                                data=image_to_bytes(out, "PNG"),
                                file_name="shear.png",
                                mime="image/png",
                            )
                        with c_jpg:
                            st.download_button(
                                "⬇ Download JPG",
                                data=image_to_bytes(out, "JPEG"),
                                file_name="shear.jpg",
                                mime="image/jpeg",
                            )

                elif mode == "reflection":
                    st.markdown(t["refl_settings"])
                    axis = st.selectbox(
                        t["refl_axis"],
                        [t["axis_x"], t["axis_y"], t["axis_diag"]],
                        key="axis_ref",
                    )
                    if st.button(f"{t['btn_apply']} ✅", key="apply_ref"):
                        h, w = original_img.shape[:2]
                        if axis == t["axis_x"]:
                            Rf = np.array([[1, 0, 0],
                                           [0, -1, h],
                                           [0, 0, 1]], dtype=np.float32)
                        elif axis == t["axis_y"]:
                            Rf = np.array([[-1, 0, w],
                                           [0, 1, 0],
                                           [0, 0, 1]], dtype=np.float32)
                        else:
                            Rf = np.array([[0, 1, 0],
                                           [1, 0, 0],
                                           [0, 0, 1]], dtype=np.float32)
                        out = apply_affine_transform(original_img, Rf)
                        st.image(out, caption=t["refl_result"], use_column_width=True)
                        c_png, c_jpg = st.columns(2)
                        with c_png:
                            st.download_button(
                                "⬇ Download PNG",
                                data=image_to_bytes(out, "PNG"),
                                file_name="reflection.png",
                                mime="image/png",
                            )
                        with c_jpg:
                            st.download_button(
                                "⬇ Download JPG",
                                data=image_to_bytes(out, "JPEG"),
                                file_name="reflection.jpg",
                                mime="image/jpeg",
                            )
                else:
                    pass

        with st.container(border=True):
            st.markdown(t["hist_title"])
            st.write(t["hist_desc"])
            show_hist = st.button(t["btn_histogram"], key="btn_histogram", type="secondary")
            if show_hist:
                if original_img is not None:
                    hist_fig = compute_histogram(original_img)
                    st.pyplot(hist_fig)
                    plt.close(hist_fig)
                else:
                    st.warning(t["hist_warning"])

    # RIGHT: Filters
    with tools_col_right:
        with st.container(border=True):
            st.markdown(t["filter_title"])
            st.write(t["filter_desc"])
            st.markdown("---")

            filter_col1, filter_col2, filter_col3 = st.columns(3)
            with filter_col1:
                if st.button(t["btn_blur"], key="btn_blur_click", type="secondary"):
                    st.session_state["image_filter"] = "blur"
            with filter_col2:
                if st.button(t["btn_sharpen"], key="btn_sharpen_click", type="secondary"):
                    st.session_state["image_filter"] = "sharpen"
            with filter_col3:
                if st.button(t["btn_background"], key="btn_bg_click", type="secondary"):
                    st.session_state["image_filter"] = "background"

            filter_col4, filter_col5, filter_col6 = st.columns(3)
            with filter_col4:
                if st.button(t["btn_grayscale"], key="btn_gray_click", type="secondary"):
                    st.session_state["image_filter"] = "grayscale"
            with filter_col5:
                if st.button(t["btn_edge"], key="btn_edge_click", type="secondary"):
                    st.session_state["image_filter"] = "edge"
            with filter_col6:
                if st.button(t["btn_brightness"], key="btn_bright_click", type="secondary"):
                    st.session_state["image_filter"] = "brightness"

        with st.container(border=True):
            if original_img is None:
                st.info(t["filter_info"])
            else:
                fmode = st.session_state["image_filter"]

                if fmode == "blur":
                    st.markdown(t["blur_settings"])
                    k = st.slider(t["blur_kernel"], 3, 31, 5, step=2, key="blur_k")
                    if st.button(f"{t['btn_apply']} ✅", key="apply_blur"):
                        img_bgr = to_opencv(original_img)
                        if k % 2 == 0:
                            k += 1
                        out_bgr = cv2.GaussianBlur(img_bgr, (k, k), 0)
                        out = to_streamlit(out_bgr)
                        st.image(out, caption=t["blur_result"], use_column_width=True)
                        c_png, c_jpg = st.columns(2)
                        with c_png:
                            st.download_button(
                                "⬇ Download PNG",
                                data=image_to_bytes(out, "PNG"),
                                file_name="blur.png",
                                mime="image/png",
                            )
                        with c_jpg:
                            st.download_button(
                                "⬇ Download JPG",
                                data=image_to_bytes(out, "JPEG"),
                                file_name="blur.jpg",
                                mime="image/jpeg",
                            )

                elif fmode == "sharpen":
                    st.markdown(t["sharpen_settings"])
                    st.write(t["sharpen_desc"])
                    if st.button(f"{t['btn_apply']} ✅", key="apply_sharp"):
                        img_bgr = to_opencv(original_img)
                        kernel = np.array([[0, -1, 0],
                                           [-1, 5, -1],
                                           [0, -1, 0]], dtype=np.float32)
                        out_bgr = cv2.filter2D(img_bgr, -1, kernel)
                        out = to_streamlit(out_bgr)
                        st.image(out, caption=t["sharpen_result"], use_column_width=True)
                        c_png, c_jpg = st.columns(2)
                        with c_png:
                            st.download_button(
                                "⬇ Download PNG",
                                data=image_to_bytes(out, "PNG"),
                                file_name="sharpen.png",
                                mime="image/png",
                            )
                        with c_jpg:
                            st.download_button(
                                "⬇ Download JPG",
                                data=image_to_bytes(out, "JPEG"),
                                file_name="sharpen.jpg",
                                mime="image/jpeg",
                            )

                elif fmode == "grayscale":
                    st.markdown(t["gray_settings"])
                    st.write(t["gray_desc"])
                    if st.button(f"{t['btn_apply']} ✅", key="apply_gray"):
                        gray = rgb_to_gray(original_img)
                        st.image(gray, caption=t["gray_result"], use_column_width=True, clamp=True)
                        c_png, c_jpg = st.columns(2)
                        with c_png:
                            st.download_button(
                                "⬇ Download PNG",
                                data=image_to_bytes(gray, "PNG"),
                                file_name="grayscale.png",
                                mime="image/png",
                            )
                        with c_jpg:
                            st.download_button(
                                "⬇ Download JPG",
                                data=image_to_bytes(gray, "JPEG"),
                                file_name="grayscale.jpg",
                                mime="image/jpeg",
                            )

                elif fmode == "edge":
                    st.markdown(t["edge_settings"])
                    method = st.selectbox(
                        t["edge_method"],
                        ["Sobel", "Canny"],
                        key="edge_method_sel",
                    )
                    if st.button(f"{t['btn_apply']} ✅", key="apply_edge"):
                        gray = rgb_to_gray(original_img)
                        if method == "Sobel":
                            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                            mag = cv2.magnitude(gx, gy)
                            out = np.clip(mag, 0, 255).astype(np.uint8)
                        else:
                            out = cv2.Canny(gray, 100, 200)
                        st.image(out, caption=t["edge_result"], use_column_width=True, clamp=True)
                        c_png, c_jpg = st.columns(2)
                        with c_png:
                            st.download_button(
                                "⬇ Download PNG",
                                data=image_to_bytes(out, "PNG"),
                                file_name="edge.png",
                                mime="image/png",
                            )
                        with c_jpg:
                            st.download_button(
                                "⬇ Download JPG",
                                data=image_to_bytes(out, "JPEG"),
                                file_name="edge.jpg",
                                mime="image/jpeg",
                            )

                elif fmode == "brightness":
                    st.markdown(t["bright_settings"])
                    b = st.slider(t["bright_brightness"], -100, 100, 0, key="bright_val")
                    c = st.slider(t["bright_contrast"], -100, 100, 0, key="contrast_val")
                    if st.button(f"{t['btn_apply']} ✅", key="apply_bright"):
                        out = adjust_brightness_contrast(original_img, brightness=b, contrast=c)
                        st.image(out, caption=t["bright_result"], use_column_width=True)
                        c_png, c_jpg = st.columns(2)
                        with c_png:
                            st.download_button(
                                "⬇ Download PNG",
                                data=image_to_bytes(out, "PNG"),
                                file_name="brightness_contrast.png",
                                mime="image/png",
                            )
                        with c_jpg:
                            st.download_button(
                                "⬇ Download JPG",
                                data=image_to_bytes(out, "JPEG"),
                                file_name="brightness_contrast.jpg",
                                mime="image/jpeg",
                            )

                elif fmode == "background":
                    st.markdown(t["bg_settings"])
                    st.write(t["bg_method"])
                    if st.button(f"{t['btn_apply']} ✅", key="apply_bg"):
                        out = simple_background_removal_hsv(original_img)
                        st.image(out, caption=t["bg_result"], use_column_width=True)
                        c_png, c_jpg = st.columns(2)
                        with c_png:
                            st.download_button(
                                "⬇ Download PNG",
                                data=image_to_bytes(out, "PNG"),
                                file_name="background_removed.png",
                                mime="image/png",
                            )
                        with c_jpg:
                            st.download_button(
                                "⬇ Download JPG",
                                data=image_to_bytes(out, "JPEG"),
                                file_name="background_removed.jpg",
                                mime="image/jpeg",
                            )

# ===================== PAGE 3: TEAM MEMBER =====================

elif page == t["nav_team"]:
    st.markdown(t["team_title"])
    st.markdown(t["team_subtitle"])

    team = [
        {
            "name": "M. khalqi riznadin",
            "sid": "004202400046",
            "role": "Leader",
            "group": "3",
            "contrib": "Project Manager, Histogram Module and UI/UX Design.",
            "photo": "images/Khalqi.jpg",
        },
        {
            "name": "Prasetya Hendramawan",
            "sid": "004202400066",
            "role": "Member",
            "group": "3",
            "contrib": "Implemented geometric transforms, Filters and Language Section.",
            "photo": "images/Pras.jpg",
        },
        {
            "name": "M. Adhiv Surya Dharma",
            "sid": "004202400004",
            "role": "Member",
            "group": "3",
            "contrib": "Designed the interface and documentation.",
            "photo": "images/Adhiv.jpg",
        },
        {
            "name": "M. Tsaqif Athallah",
            "sid": "004202400022",
            "role": "Member",
            "group": "3",
            "contrib": "Designed the app concept and overall workflow.",
            "photo": "images/Tsaqif.jpg",
        },
    ]

    cols = st.columns(4)
    for col, member in zip(cols, team):
        with col:
            st.markdown(f"<h4 style='text-align: center;'>{member['name']}</h4>", unsafe_allow_html=True)
            if os.path.exists(member["photo"]):
                img = Image.open(member["photo"]).convert("RGB")
                w, h = img.size
                m = min(w, h)
                left = (w - m) // 2
                top = (h - m) // 2
                img = img.crop((left, top, left + m, top + m))
                img = img.resize((140, 140), Image.Resampling.LANCZOS)
                st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
                st.image(img, use_column_width=False)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    """
                    <div class="team-photo-container">
                        <div style="width:100%; height:100%; display:flex; align-items:center; justify-content:center; background:#ddd; color:#666;">
                            No Image
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.write(f"{t['team_sid']} {member['sid']}")
            st.write(f"{t['team_role']} {member['role']}")
            st.write(f"{t['team_group']} {member['group']}")
            st.write(f"{t['team_contribution']} {member['contrib']}")
