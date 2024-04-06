import torch
import torch.nn as nn
import numpy as np


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, 1, 1)

    def forward(self, x):
        return self.conv(x)

    def tile_forward(self, x, patch_size):
        _, _, h, w = x.size()
        p_h, p_w = patch_size

        n_block_h = int(np.ceil(h / p_h))
        n_block_w = int(np.ceil(w / p_w))

        output = torch.zeros_like(x)
        for i in range(n_block_h):
            for j in range(n_block_w):
                anchor, anchor_rd, anchor_pad, anchor_rd_pad = self._get_patch(
                    x, [i * p_h, j * p_w], patch_size, [h, w]
                )
                output[
                    :, :, anchor[0] : anchor_rd[0], anchor[1] : anchor_rd[1]
                ] = self.forward(
                    x[
                        :,
                        :,
                        anchor_pad[0] : anchor_rd_pad[0],
                        anchor_pad[1] : anchor_rd_pad[1],
                    ]
                )[
                    :,
                    :,
                    anchor[0] - anchor_pad[0] : anchor_rd[0] - anchor_pad[0],
                    anchor[1] - anchor_pad[1] : anchor_rd[1] - anchor_pad[1],
                ]

        return output

    def _get_patch(self, img, anchor, patch_size, img_size, pad=2):
        anchor_rd = [anchor[0] + patch_size[0], anchor[1] + patch_size[1]]
        anchor_rd[0] = np.clip(anchor_rd[0], 0, img_size[0])
        anchor_rd[1] = np.clip(anchor_rd[1], 0, img_size[1])

        anchor, anchor_rd = np.asarray(anchor), np.asarray(anchor_rd)
        anchor_pad = np.clip(anchor - pad, 0, max(img_size[0], img_size[1]))
        anchor_rd_pad = anchor_rd + pad
        anchor_rd_pad[0] = np.clip(anchor_rd_pad[0], 0, img_size[0])
        anchor_rd_pad[1] = np.clip(anchor_rd_pad[1], 0, img_size[1])

        return anchor, anchor_rd, anchor_pad, anchor_rd_pad


if __name__ == "__main__":

    net = Net()
    x = torch.rand(1, 1, 20, 20)
    y_crop = net.tile_forward(x, [8, 8])
    y_full = net(x)
    print(torch.sum(y_crop - y_full))
