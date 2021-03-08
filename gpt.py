class ModelSettings:
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(
        self, size: str, n_layer: int, d_model: int, learning_rate: float,
    ):
        self.size = size
        self.n_layer = n_layer
        self.d_model = d_model
        self.learning_rate = learning_rate
        self.n_head = max(2, self.d_model // 64)
        self.d_ff = 4 * d_model
        self.d_attn = 1 * d_model


common_models_by_name = {
    "x10small": ModelSettings(
        size="x10small", n_layer=1, d_model=8, learning_rate=0.00211,
    ),
    "x9small": ModelSettings(
        size="x9small", n_layer=1, d_model=16, learning_rate=0.00211,
    ),
    "x8small": ModelSettings(
        size="x8small", n_layer=1, d_model=32, learning_rate=0.00211,
    ),
    "x7small": ModelSettings(
        size="x7small", n_layer=2, d_model=32, learning_rate=0.00211,
    ),
    "x6small": ModelSettings(
        size="x6small", n_layer=2, d_model=64, learning_rate=0.00211,
    ),
    "x5small": ModelSettings(
        size="x5small", n_layer=2, d_model=128, learning_rate=0.00202,
    ),
    "x4small": ModelSettings(
        size="x4small", n_layer=4, d_model=256, learning_rate=0.00173,
    ),
    "x3small": ModelSettings(
        size="x3small", n_layer=4, d_model=512, learning_rate=0.00163,
    ),
    "x2small": ModelSettings(
        size="x2small", n_layer=8, d_model=512, learning_rate=0.00144,
    ),
    "x1small": ModelSettings(
        size="x1small", n_layer=6, d_model=768, learning_rate=0.00146,
    ),
    "small": ModelSettings(
        size="small", n_layer=12, d_model=768, learning_rate=0.0006,
    ),
    "medium": ModelSettings(
        size="medium", n_layer=24, d_model=1024, learning_rate=0.0003,
    ),
    "large": ModelSettings(
        size="large", n_layer=24, d_model=1536, learning_rate=0.00025,
    ),
    "xl": ModelSettings(size="xl", n_layer=24, d_model=2048, learning_rate=0.00000625,),
}
