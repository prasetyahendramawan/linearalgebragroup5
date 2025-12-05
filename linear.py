# Streamlit app: 2D Matrix Transformations (Homogeneous Coordinates)
# English + Indonesian version with enhanced PDF generation (colors, layout, explanations) for PDF (A4 full page), PNG, JPG, and CSV

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io

# --- Transformation matrix functions (3x3 homogeneous coordinates) ---

def translation_matrix(tx, ty):
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])


def scaling_matrix(sx, sy):
    return np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0, 0, 1]
    ])


def rotation_matrix(theta_deg):
    theta = np.radians(theta_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return np.array([
        [cos_t, -sin_t, 0],
        [sin_t, cos_t, 0],
        [0, 0, 1]
    ])


def shearing_matrix(shx, shy):
    return np.array([
        [1, shx, 0],
        [shy, 1, 0],
        [0, 0, 1]
    ])


def reflection_matrix(kind):
    if kind == "X-axis":
        return np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
    elif kind == "Y-axis":
        return np.array([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
    elif kind == "Line y = x":
        return np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
    elif kind == "Origin":
        return np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]
        ])
    else:
        return np.identity(3)


# --- Base shape: triangle in homogeneous coordinates (x, y, 1) ---
base_shape = np.array([
    [0, 0, 1],
    [1, 0, 1],
    [0.5, 1, 1],
    [0, 0, 1]
]).T  # shape 3x4


# --- Plot helper ---
def plot_shapes(original, transformed, title_left="Original Shape", title_right="Transformed Shape", fig_size=(10,5)):
    fig, axes = plt.subplots(1, 2, figsize=fig_size)
    axes[0].plot(original[0, :], original[1, :], "o-", color="#10B981")
    axes[0].set_title(title_left)
    axes[0].axis('equal')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    axes[1].plot(transformed[0, :], transformed[1, :], "o-", color="orange")
    axes[1].set_title(title_right)
    axes[1].axis('equal')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    for ax in axes:
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)

    plt.tight_layout()
    return fig


# --- Compose transformations according to user-specified order ---
def compose_transformations(order, params):
    mat_dict = {
        "Translation": translation_matrix(params['tx'], params['ty']),
        "Scaling": scaling_matrix(params['sx'], params['sy']),
        "Rotation": rotation_matrix(params['theta']),
        "Shearing": shearing_matrix(params['shx'], params['shy']),
        "Reflection": reflection_matrix(params['reflection'])
    }
    composite = np.identity(3)
    for name in order:
        composite = mat_dict[name] @ composite
    return composite, mat_dict


# --- Streamlit UI ---
st.set_page_config(page_title="2D Matrix Transformations (Homogeneous Coordinates)", layout="wide")
st.title("Interactive 2D Matrix Transformations (Homogeneous Coordinates)")
st.markdown(
    """
    This app demonstrates how 2D transformations — translation, scaling, rotation, shearing, and reflection — can be represented
    as 3x3 matrices in homogeneous coordinates. Using homogeneous coordinates allows all transforms to be composed by matrix
    multiplication.
    """
)

# Sidebar inputs
st.sidebar.header("Transformation Settings")

tx = st.sidebar.slider("Translation: tx", -2.0, 2.0, 0.0, 0.1)
ty = st.sidebar.slider("Translation: ty", -2.0, 2.0, 0.0, 0.1)

sx = st.sidebar.slider("Scaling: sx", 0.1, 3.0, 1.0, 0.1)
sy = st.sidebar.slider("Scaling: sy", 0.1, 3.0, 1.0, 0.1)

theta = st.sidebar.slider("Rotation: θ (degrees)", -180, 180, 0, 1)

shx = st.sidebar.slider("Shearing: shx", -2.0, 2.0, 0.0, 0.1)
shy = st.sidebar.slider("Shearing: shy", -2.0, 2.0, 0.0, 0.1)

reflection_type = st.sidebar.selectbox("Reflect about:", ("None", "X-axis", "Y-axis", "Line y = x", "Origin"))

st.sidebar.markdown("---")
st.sidebar.subheader("Composition Order (choose order from top to bottom)")
default_order = ["Translation", "Rotation", "Scaling"]
order = st.sidebar.multiselect(
    "Select and order transformations (multiselect preserves your chosen order)",
    options=["Translation", "Scaling", "Rotation", "Shearing", "Reflection"],
    default=default_order
)

# If reflection type is set but not in order, append it; warn if mismatch
if "Reflection" in order and reflection_type == "None":
    st.sidebar.warning("You included Reflection in the order but selected 'None' for reflection type.")
elif "Reflection" not in order and reflection_type != "None":
    order.append("Reflection")

params = {
    'tx': tx,
    'ty': ty,
    'sx': sx,
    'sy': sy,
    'theta': theta,
    'shx': shx,
    'shy': shy,
    'reflection': reflection_type if reflection_type != "None" else None
}

# Compute composite and transformed shape
composite_matrix, matrices = compose_transformations(order, params)
transformed_shape = composite_matrix @ base_shape

# Show plot
fig = plot_shapes(base_shape, transformed_shape)
st.pyplot(fig)

# Show matrices
st.subheader("3x3 Transformation Matrices")
for name in order:
    st.markdown(f"**{name}**")
    st.write(matrices[name])

st.markdown("*Composite matrix (product of all transformations in the chosen order):*")
st.write(composite_matrix)

# Data table of points
st.subheader("Point Coordinates Table")

n_points = base_shape.shape[1]
data = []
for i in range(n_points):
    x, y = base_shape[0, i], base_shape[1, i]
    x_t, y_t = transformed_shape[0, i], transformed_shape[1, i]
    data.append([x, y, x_t, y_t])

df = pd.DataFrame(data, columns=["x", "y", "x' (result)", "y' (result)"])
st.dataframe(df)

# --- Download utilities ---

def fig_to_bytes(fig, fmt='png', dpi=300, a4=False):
    buf = io.BytesIO()
    # Optionally resize to A4 for PDF
    if a4:
        # Save current size, set to A4, save, then restore
        orig_size = fig.get_size_inches()
        fig.set_size_inches(8.27, 11.69)
        fig.savefig(buf, format=fmt if fmt!='jpg' else 'jpeg', dpi=dpi, bbox_inches='tight')
        fig.set_size_inches(orig_size)
    else:
        fig.savefig(buf, format=fmt if fmt!='jpg' else 'jpeg', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return buf


def df_to_csv_bytes(dataframe):
    buf = io.StringIO()
    dataframe.to_csv(buf, index=False)
    buf.seek(0)
    return buf.getvalue().encode('utf-8')

# Download buttons
st.markdown("---")
st.subheader("Download Results")

# CSV (points)
csv_bytes = df_to_csv_bytes(df)
st.download_button(label="Download points as CSV", data=csv_bytes, file_name="points_transformed.csv", mime="text/csv")

# Composite matrix CSV
comp_df = pd.DataFrame(composite_matrix)
comp_csv = df_to_csv_bytes(comp_df)
st.download_button(label="Download composite matrix (CSV)", data=comp_csv, file_name="composite_matrix.csv", mime="text/csv")

# PNG
png_bytes = fig_to_bytes(fig, fmt='png', dpi=300, a4=False)
st.download_button(label="Download plot as PNG", data=png_bytes, file_name="transformation_plot.png", mime="image/png")

# JPG (JPEG)
jpg_bytes = fig_to_bytes(fig, fmt='jpg', dpi=300, a4=False)
st.download_button(label="Download plot as JPG", data=jpg_bytes, file_name="transformation_plot.jpg", mime="image/jpeg")

# PDF (A4 full page)
# We'll save the same plot but resize to A4 for a full-page PDF
pdf_bytes = fig_to_bytes(fig, fmt='pdf', dpi=300, a4=True)
st.download_button(label="Download plot as PDF (A4)", data=pdf_bytes, file_name="transformation_plot_A4.pdf", mime="application/pdf")

st.markdown("""

---
Tips: Use the 'Composition Order' to change how transformations are applied. The order matters: matrix multiplication is not commutative.
""")
