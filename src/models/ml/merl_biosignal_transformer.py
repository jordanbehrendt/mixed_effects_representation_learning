import torch
import torch.nn as nn


class MERLBiosignalTransformer(nn.Module):
    """
    CNN + Transformer regression model for multimodal biosignals.

    Expected input:
        x: (B, T, 5)

    The five channels are assumed to be:
        0: ECG
        1: PPG
        2: Respiration
        3: EMG1
        4: EMG2

    Optional extra features:
        extra_features: (B, extra_dim)

    Output:
        y: (B,)

    Architecture:
        ECG      ──┐
        PPG      ──┤
        Resp     ──┤
        EMG1     ──┤→ modality-specific CNNs
        EMG2     ──┘
                    ↓
                 Fusion
                    ↓
              Downsampling
                    ↓
               Transformer
                    ↓
                 Pooling
                    ↓
             + extra features
                    ↓
                  MLP
                    ↓
                regression

    If ``use_signal_branch=False``, the entire CNN + fusion + transformer
    pipeline is skipped, and the model becomes a plain MLP regressor on
    ``extra_features`` alone. This requires ``extra_dim > 0``. Useful for
    ablations (e.g. "does the biosignal branch actually help over just
    tabular features?").
    """

    def __init__(
        self,
        input_dim=5,
        extra_dim=0,
        d_model=128,
        n_heads=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        activation="gelu",
        pooling="cls",
        max_seq_len=2000,
        conv_channels=32,
        conv_kernel_size=9,
        downsample_factor=2,
        use_signal_branch=True,
    ):
        super().__init__()

        if input_dim != 5:
            raise ValueError(
                f"This model expects exactly 5 biosignal channels "
                f"(ECG, PPG, Resp, EMG1, EMG2), got {input_dim}."
            )

        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by "
                f"n_heads ({n_heads})."
            )

        if pooling not in {"mean", "max", "cls"}:
            raise ValueError(
                f"Unknown pooling method '{pooling}'. "
                f"Use 'mean', 'max', or 'cls'."
            )

        if not use_signal_branch and extra_dim <= 0:
            raise ValueError(
                "use_signal_branch=False requires extra_dim > 0, since "
                "the model would otherwise have no input to regress on."
            )

        self.input_dim = input_dim
        self.extra_dim = extra_dim
        self.d_model = d_model
        self.pooling = pooling
        self.max_seq_len = max_seq_len
        self.downsample_factor = downsample_factor
        self.use_signal_branch = use_signal_branch

        if self.use_signal_branch:
            # ============================================================
            # 1. Modality-specific temporal encoders
            # ============================================================
            #
            # Each physiological signal gets its own filters because ECG,
            # PPG, respiration and EMG have very different characteristics.
            #

            def make_modality_encoder():
                return nn.Sequential(
                    nn.Conv1d(
                        in_channels=1,
                        out_channels=conv_channels,
                        kernel_size=conv_kernel_size,
                        padding=conv_kernel_size // 2,
                    ),
                    nn.BatchNorm1d(conv_channels),
                    nn.GELU(),

                    nn.Conv1d(
                        in_channels=conv_channels,
                        out_channels=conv_channels,
                        kernel_size=conv_kernel_size,
                        padding=conv_kernel_size // 2,
                    ),
                    nn.BatchNorm1d(conv_channels),
                    nn.GELU(),
                )

            self.ecg_encoder = make_modality_encoder()
            self.ppg_encoder = make_modality_encoder()
            self.resp_encoder = make_modality_encoder()
            self.emg1_encoder = make_modality_encoder()
            self.emg2_encoder = make_modality_encoder()

            # Five modalities × conv_channels
            fused_dim = 5 * conv_channels

            # ============================================================
            # 2. Fusion + temporal downsampling
            # ============================================================

            self.fusion = nn.Sequential(
                nn.Conv1d(
                    in_channels=fused_dim,
                    out_channels=d_model,
                    kernel_size=1,
                ),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
            )

            # Reduce sequence length before self-attention.
            #
            # Example:
            #     T = 2000
            #     downsample_factor = 2
            #     → T = 1000
            #
            # This makes the Transformer substantially cheaper.
            if downsample_factor > 1:
                self.downsampler = nn.Conv1d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=downsample_factor,
                    stride=downsample_factor,
                )
            else:
                self.downsampler = nn.Identity()

            # ============================================================
            # 3. CLS token
            # ============================================================

            if pooling == "cls":
                self.cls_token = nn.Parameter(
                    torch.randn(1, 1, d_model) * 0.02
                )

            # ============================================================
            # 4. Positional embedding
            # ============================================================

            max_transformer_len = (
                max_seq_len // downsample_factor
                if downsample_factor > 1
                else max_seq_len
            )

            if pooling == "cls":
                max_transformer_len += 1

            self.positional_embedding = nn.Parameter(
                torch.randn(
                    1,
                    max_transformer_len,
                    d_model,
                ) * 0.02
            )

            # ============================================================
            # 5. Transformer encoder
            # ============================================================

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=True,
                norm_first=True,
            )

            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers,
            )

            self.transformer_norm = nn.LayerNorm(d_model)

        # ============================================================
        # 6. Regression head
        # ============================================================

        regressor_input_dim = (
            (d_model + extra_dim) if self.use_signal_branch else extra_dim
        )
        regressor_hidden_dim = max(regressor_input_dim // 2, 1)

        self.regressor = nn.Sequential(
            nn.Linear(
                regressor_input_dim,
                regressor_hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(
                regressor_hidden_dim,
                1,
            ),
        )

    def forward(self, x=None, extra_features=None):
        """
        Parameters
        ----------
        x : torch.Tensor, optional
            Shape: (B, T, 5)

            Channels:
                0 = ECG
                1 = PPG
                2 = Respiration
                3 = EMG1
                4 = EMG2

            Ignored (may be None) if ``use_signal_branch=False``.

        extra_features : torch.Tensor, optional
            Shape: (B, extra_dim)

        Returns
        -------
        torch.Tensor
            Shape: (B,)
        """

        # ------------------------------------------------------------
        # Bypass mode: extra_features-only MLP
        # ------------------------------------------------------------

        if not self.use_signal_branch:
            if extra_features is None:
                raise ValueError(
                    "Model was initialized with use_signal_branch=False, "
                    "but extra_features was None."
                )

            if extra_features.ndim != 2:
                raise ValueError(
                    "extra_features must have shape "
                    "(B, extra_dim)."
                )

            if extra_features.shape[1] != self.extra_dim:
                raise ValueError(
                    f"Expected {self.extra_dim} extra features, "
                    f"got {extra_features.shape[1]}."
                )

            out = self.regressor(extra_features)
            return out.squeeze(-1)

        # ------------------------------------------------------------
        # Full signal branch
        # ------------------------------------------------------------

        if x is None:
            raise ValueError(
                "x must be provided when use_signal_branch=True."
            )

        if x.ndim != 3:
            raise ValueError(
                f"Expected x to have shape (B, T, 5), "
                f"got {tuple(x.shape)}."
            )

        B, T, F = x.shape

        if F != 5:
            raise ValueError(
                f"Expected 5 input features, got {F}."
            )

        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds max_seq_len "
                f"{self.max_seq_len}."
            )

        # ------------------------------------------------------------
        # Convert:
        #   (B, T, 5)
        #
        # to individual modality tensors:
        #   (B, 1, T)
        # ------------------------------------------------------------

        ecg = x[:, :, 0].unsqueeze(1)
        ppg = x[:, :, 1].unsqueeze(1)
        resp = x[:, :, 2].unsqueeze(1)
        emg1 = x[:, :, 3].unsqueeze(1)
        emg2 = x[:, :, 4].unsqueeze(1)

        # ------------------------------------------------------------
        # Modality-specific CNN processing
        # ------------------------------------------------------------

        ecg = self.ecg_encoder(ecg)
        ppg = self.ppg_encoder(ppg)
        resp = self.resp_encoder(resp)
        emg1 = self.emg1_encoder(emg1)
        emg2 = self.emg2_encoder(emg2)

        # Each:
        #   (B, conv_channels, T)
        #
        # Concatenate:
        #   (B, 5 * conv_channels, T)

        x = torch.cat(
            [
                ecg,
                ppg,
                resp,
                emg1,
                emg2,
            ],
            dim=1,
        )

        # ------------------------------------------------------------
        # Fusion
        # ------------------------------------------------------------

        x = self.fusion(x)

        # Shape:
        #   (B, d_model, T)

        # ------------------------------------------------------------
        # Downsample temporal dimension
        # ------------------------------------------------------------

        x = self.downsampler(x)

        # Shape:
        #   (B, d_model, T_reduced)

        # Transformer expects:
        #   (B, T, d_model)

        x = x.transpose(1, 2)

        B, T_reduced, _ = x.shape

        # ------------------------------------------------------------
        # CLS token
        # ------------------------------------------------------------

        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(
                B,
                -1,
                -1,
            )

            x = torch.cat(
                [
                    cls_tokens,
                    x,
                ],
                dim=1,
            )

            T_reduced += 1

        # ------------------------------------------------------------
        # Positional encoding
        # ------------------------------------------------------------

        if T_reduced > self.positional_embedding.shape[1]:
            raise ValueError(
                f"Transformer sequence length {T_reduced} exceeds "
                f"available positional embeddings "
                f"{self.positional_embedding.shape[1]}."
            )

        x = (
            x
            + self.positional_embedding[:, :T_reduced]
        )

        # ------------------------------------------------------------
        # Transformer
        # ------------------------------------------------------------

        x = self.transformer(x)

        x = self.transformer_norm(x)

        # ------------------------------------------------------------
        # Pooling
        # ------------------------------------------------------------

        if self.pooling == "mean":

            x = x.mean(dim=1)

        elif self.pooling == "max":

            x = x.max(dim=1).values

        elif self.pooling == "cls":

            x = x[:, 0]

        # x:
        #   (B, d_model)

        # ------------------------------------------------------------
        # Extra features
        # ------------------------------------------------------------

        if self.extra_dim > 0:

            if extra_features is None:
                raise ValueError(
                    f"Model was initialized with extra_dim="
                    f"{self.extra_dim}, but extra_features was None."
                )

            if extra_features.ndim != 2:
                raise ValueError(
                    "extra_features must have shape "
                    "(B, extra_dim)."
                )

            if extra_features.shape[0] != B:
                raise ValueError(
                    "Batch size of extra_features does not "
                    "match x."
                )

            if extra_features.shape[1] != self.extra_dim:
                raise ValueError(
                    f"Expected {self.extra_dim} extra features, "
                    f"got {extra_features.shape[1]}."
                )

            x = torch.cat(
                [
                    x,
                    extra_features,
                ],
                dim=-1,
            )

        elif extra_features is not None:
            raise ValueError(
                "extra_features were provided, but extra_dim=0."
            )

        # ------------------------------------------------------------
        # Regression
        # ------------------------------------------------------------

        x = self.regressor(x)

        return x.squeeze(-1)