"""
Ejercicio 1 - Procesamiento de Imágenes: Road Sign Detection
============================================================
Dataset: Road Sign Detection
Imágenes: road0, road1, road103, road105, road106, road108,
          road109, road111, road112, road116

Requisitos:
    pip install opencv-python numpy matplotlib

Coloca las 10 imágenes en la misma carpeta que este script,
o ajusta la variable IMAGE_DIR con la ruta correcta.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ──────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────
IMAGE_DIR = "."   # Carpeta donde están las imágenes
IMAGE_NAMES = [
    "road5.png", "road461.png", "road191.png", "road6.png", "road43.png", "road44.png", "road162.png",
    "road212.png", "road260.png", "road293.png"

]

GRAY_THRESHOLD   = 127   # Umbral para binarización en grises (0-255)
COLOR_CHANNEL    = 2     # Canal para binarización: 0=B, 1=G, 2=R
CHANNEL_THRESHOLD = 100  # Umbral sobre el canal de color

# Niveles de ruido Gaussiano (desviación estándar)
NOISE_LEVELS = [10, 25, 50]


# ──────────────────────────────────────────────
# UTILIDADES
# ──────────────────────────────────────────────

def load_images(image_dir, names):
    """Carga imágenes en BGR; omite las que no existan."""
    images, loaded = [], []
    for name in names:
        path = os.path.join(image_dir, name)
        img = cv2.imread(path)
        if img is None:
            print(f"  [ADVERTENCIA] No se encontró: {path}")
            continue
        images.append(img)
        loaded.append(name)
    print(f"  {len(images)} imágenes cargadas: {loaded}")
    return images, loaded


def add_gaussian_noise(gray, sigma):
    """Agrega ruido Gaussiano a una imagen en escala de grises."""
    noise = np.random.normal(0, sigma, gray.shape).astype(np.float32)
    noisy = np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def fig_title(fig, title, fontsize=13):
    fig.suptitle(title, fontsize=fontsize, fontweight="bold", y=1.01)


# ──────────────────────────────────────────────
# PARTE 1 — Color y Escala de Grises
# ──────────────────────────────────────────────

def parte1_color_gris(images, names):
    print("\n[PARTE 1] Color vs. Escala de Grises")
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(n * 2.2, 5))
    fig_title(fig, "Parte 1 — Imágenes a Color y en Escala de Grises")

    for i, (img, name) in enumerate(zip(images, names)):
        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        axes[0, i].imshow(rgb)
        axes[0, i].set_title(name, fontsize=7)
        axes[0, i].axis("off")

        axes[1, i].imshow(gray, cmap="gray")
        axes[1, i].set_title("Gris", fontsize=7)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Color", fontsize=9)
    axes[1, 0].set_ylabel("Gris", fontsize=9)
    plt.tight_layout()
    plt.savefig("parte1_color_gris.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("  → Guardado: parte1_color_gris.png")


# ──────────────────────────────────────────────
# PARTE 2 — Binarización por Umbral en Grises
# ──────────────────────────────────────────────

def parte2_binarizacion_gris(images, names, threshold=127):
    print(f"\n[PARTE 2] Binarización en Grises (umbral={threshold})")
    n = len(images)
    fig, axes = plt.subplots(3, n, figsize=(n * 2.2, 7))
    fig_title(
        fig,
        f"Parte 2 — Color / Gris / Binaria (umbral gris={threshold})"
    )

    for i, (img, name) in enumerate(zip(images, names)):
        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        axes[0, i].imshow(rgb);           axes[0, i].axis("off")
        axes[0, i].set_title(name, fontsize=6)
        axes[1, i].imshow(gray, cmap="gray"); axes[1, i].axis("off")
        axes[2, i].imshow(binary, cmap="gray"); axes[2, i].axis("off")

    axes[0, 0].set_ylabel("Color",   fontsize=9)
    axes[1, 0].set_ylabel("Grises",  fontsize=9)
    axes[2, 0].set_ylabel("Binaria", fontsize=9)
    plt.tight_layout()
    plt.savefig("parte2_binaria_gris.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("  → Guardado: parte2_binaria_gris.png")

    # Análisis textual
    print("""
  ── Análisis Parte 2 ─────────────────────────────────────────────────────
  • Imagen a COLOR: La más informativa para identificar el TIPO de señal.
    El color es discriminante: rojo → alto/peligro, verde → paso permitido,
    amarillo → precaución. Para semáforos, el canal rojo/verde es clave.
  • Imagen en GRISES: Útil para detectar forma y texto (números de límite
    de velocidad, silueta de la señal). Pierde información cromática.
  • Imagen BINARIA (umbral gris): Separa fondo claro de objetos oscuros
    o viceversa. Buena para detectar bordes y formas simples, pero sensible
    a la iluminación. Para señales de velocidad, resalta los dígitos si el
    umbral es adecuado; para semáforos, pierde la distinción de color.
  ─────────────────────────────────────────────────────────────────────────
""")


# ──────────────────────────────────────────────
# PARTE 3 — Binarización por Canal de Color
# ──────────────────────────────────────────────

def parte3_binarizacion_canal(images, names, channel=2, threshold=100):
    channel_names = {0: "Azul (B)", 1: "Verde (G)", 2: "Rojo (R)"}
    ch_label = channel_names.get(channel, str(channel))
    print(f"\n[PARTE 3] Binarización por canal {ch_label} (umbral={threshold})")

    n = len(images)
    fig, axes = plt.subplots(3, n, figsize=(n * 2.2, 7))
    fig_title(
        fig,
        f"Parte 3 — Color / Canal {ch_label} / Binaria (umbral={threshold})"
    )

    for i, (img, name) in enumerate(zip(images, names)):
        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ch   = img[:, :, channel]   # img está en BGR
        _, binary = cv2.threshold(ch, threshold, 255, cv2.THRESH_BINARY)

        axes[0, i].imshow(rgb);           axes[0, i].axis("off")
        axes[0, i].set_title(name, fontsize=6)
        axes[1, i].imshow(ch, cmap="gray"); axes[1, i].axis("off")
        axes[2, i].imshow(binary, cmap="gray"); axes[2, i].axis("off")

    axes[0, 0].set_ylabel("Color",            fontsize=9)
    axes[1, 0].set_ylabel(f"Canal {ch_label}", fontsize=9)
    axes[2, 0].set_ylabel("Binaria",           fontsize=9)
    plt.tight_layout()
    plt.savefig("parte3_binaria_canal.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("  → Guardado: parte3_binaria_canal.png")

    print(f"""
  ── Análisis Parte 3 ─────────────────────────────────────────────────────
  Canal usado: {ch_label}
  • Binarizar sobre el canal ROJO permite aislar señales y semáforos con
    componente roja dominante (señales de stop, luz roja, señales de límite
    de velocidad con borde rojo). El canal rojo separa mejor los objetos
    rojizos del fondo verde/azul del entorno urbano.
  • Comparado con la binarización en grises (Parte 2), la binarización por
    canal de color SÍ resulta más informativa cuando el color es el rasgo
    discriminante (p.ej. semáforo rojo vs verde). Permite segmentar
    regiones de interés que en grises se confunden con el fondo.
  • Para señales de límite de velocidad (borde rojo, interior blanco) el
    canal R resalta el contorno circular de forma más limpia.
  ─────────────────────────────────────────────────────────────────────────
""")


# ──────────────────────────────────────────────
# PARTE 4 — Ruido Gaussiano y Filtros de Suavizado
# ──────────────────────────────────────────────

def parte4_ruido_y_filtros(images, names, noise_levels=None):
    if noise_levels is None:
        noise_levels = [10, 25, 50]
    print(f"\n[PARTE 4] Ruido Gaussiano + Filtros (σ={noise_levels})")

    # Mostramos las 3 primeras imágenes para no saturar la figura
    demo_imgs  = images[:3]
    demo_names = names[:3]

    filtros = {
        "Gaussiano 5×5": lambda g: cv2.GaussianBlur(g, (5, 5), 0),
        "Media 5×5":     lambda g: cv2.blur(g, (5, 5)),
        "Mediana 5":     lambda g: cv2.medianBlur(g, 5),
        "Bilateral":     lambda g: cv2.bilateralFilter(g, 9, 75, 75),
    }

    for sigma in noise_levels:
        n_rows = len(demo_imgs)
        n_cols = 2 + len(filtros)   # original + ruidosa + filtros
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5)
        )
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        fig_title(
            fig,
            f"Parte 4 — Ruido Gaussiano σ={sigma} y Filtros de Suavizado"
        )

        col_titles = ["Original", f"+ Ruido σ={sigma}"] + list(filtros.keys())
        for c, t in enumerate(col_titles):
            axes[0, c].set_title(t, fontsize=8, fontweight="bold")

        for r, (img, name) in enumerate(zip(demo_imgs, demo_names)):
            gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            noisy = add_gaussian_noise(gray, sigma)

            axes[r, 0].imshow(gray,  cmap="gray"); axes[r, 0].axis("off")
            axes[r, 1].imshow(noisy, cmap="gray"); axes[r, 1].axis("off")
            axes[r, 0].set_ylabel(name, fontsize=7)

            for c, fn in enumerate(filtros.values(), start=2):
                filtered = fn(noisy)
                axes[r, c].imshow(filtered, cmap="gray")
                axes[r, c].axis("off")

        plt.tight_layout()
        fname = f"parte4_ruido_sigma{sigma}.png"
        plt.savefig(fname, dpi=120, bbox_inches="tight")
        plt.show()
        print(f"  → Guardado: {fname}")

    print("""
  ── Análisis Parte 4 ─────────────────────────────────────────────────────
  • Filtro de Media: Reduce ruido pero difumina bordes. Adecuado para ruido
    bajo (σ ≤ 10). Con ruido alto pierde detalles importantes de la señal.
  • Filtro Gaussiano: Similar a la media pero pesa más el centro; mejor
    preservación de bordes suaves. Buen balance general.
  • Filtro de Mediana: Excelente contra ruido impulsivo; aquí elimina picos
    del Gaussiano sin borrar bordes abruptos. MUY RECOMENDADO para señales
    de tránsito donde los bordes y los dígitos son críticos.
  • Filtro Bilateral: Preserva bordes nítidos mientras suaviza regiones
    uniformes. Es el más costoso computacionalmente pero el de mejor calidad
    perceptual. IDEAL cuando se necesita mantener el texto/números legibles.

  RECOMENDACIÓN: Para señales de tránsito usar MEDIANA (bajo σ) o
  BILATERAL (σ alto) ya que ambos respetan los contornos discriminantes.
  ─────────────────────────────────────────────────────────────────────────
""")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Ejercicio 1 — Road Sign Detection: Procesamiento de Imágenes")
    print("=" * 60)

    images, names = load_images(IMAGE_DIR, IMAGE_NAMES)
    if not images:
        print("\n  ERROR: No se cargó ninguna imagen. Revisa IMAGE_DIR.")
        return

    parte1_color_gris(images, names)
    parte2_binarizacion_gris(images, names, threshold=GRAY_THRESHOLD)
    parte3_binarizacion_canal(images, names, channel=COLOR_CHANNEL,
                              threshold=CHANNEL_THRESHOLD)
    parte4_ruido_y_filtros(images, names, noise_levels=NOISE_LEVELS)

    print("\n  ✓ Ejercicio completado. Todas las figuras han sido guardadas.")


if __name__ == "__main__":
    main()
