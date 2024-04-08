import numpy as np
from torch import Tensor
import matplotlib.pyplot as plt

from s_multimae.model.multimae import build_2d_sincos_posemb


def visualize_2d_posemb():
    NH, NW = 14, 14
    dim_tokens = 768

    colors = [
        "Greys",
        "Purples",
        "Blues",
        "Greens",
        "Oranges",
        "Reds",
        "YlOrBr",
        "YlOrRd",
        "OrRd",
        "PuRd",
        "RdPu",
        "BuPu",
        "GnBu",
        "PuBu",
        "YlGnBu",
        "PuBuGn",
        "BuGn",
        "YlGn",
    ]

    pos_emb: Tensor = build_2d_sincos_posemb(NH, NW, dim_tokens)
    pos_emb_numpy: np.ndarray = (
        pos_emb.squeeze(0).permute(1, 2, 0).numpy()
    )  # 14 x 14 x 768

    x = np.linspace(0, NH - 1, NH)
    y = np.linspace(0, NW - 1, NW)
    X, Y = np.meshgrid(x, y)

    for color, i in zip(colors, range(0, pos_emb_numpy.shape[2], 100)):
        ax = plt.axes(projection="3d")
        Z = pos_emb_numpy[:, :, i]

        # plt.imshow(Z, cmap='viridis')
        # plt.savefig(f'posemb_visualization/test_{i}.png')

        ax.plot_surface(
            X,
            Y,
            Z,
            # rstride=1, cstride=1,
            cmap="viridis",
            edgecolor="none",
        )
        plt.show()
        plt.savefig(f"posemb_visualization/test_{i}.png")
