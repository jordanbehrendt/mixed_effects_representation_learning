import numpy as np
import torch

from src.utils.torch_helpers import (
    get_device,
    get_loss,
    get_optimizer,
)


class TorchModelWrapper:

    def __init__(self, model_fn, config, model_name="torch_model"):
        """Initialize wrapper with model factory and config."""

        self.model_fn = model_fn
        self.config = config
        self.device = get_device(self.config)

        self.model = None
        self.loss_fn = None
        self.optimizer = None

        self.model_name = model_name
        self.best_loss = float("inf")
        self.best_state = None
        self.patience_counter = 0

    def _build_model(self, input_dim, config):
        """Create model instance from factory function."""
        return self.model_fn(input_dim, config).to(self.device)

    def _setup(self, input_dim, config):
        """Initialize model, loss function, and optimizer."""

        if self.model is None:
            self.model = self._build_model(input_dim, config)

        self.loss_fn = get_loss(config.get("loss", "mse"))

        opt_cfg = config.get("optimizer", {"name": "adam", "lr": 1e-3})

        self.optimizer = get_optimizer(
            opt_cfg.get("name", "adam"),
            {k: v for k, v in opt_cfg.items() if k != "name"},  # remove "name" key before passing
            self.model.parameters(),
        )

        self.epochs = config.get("epochs", 50)
        self.batch_size = config.get("batch_size", 32)
        self.patience = config.get("patience", 5)

    def _train_step(self, xb, yb, extra=None):
        """Perform a single training update step."""

        self.optimizer.zero_grad()  # reset gradients

        preds = self.model(xb, extra)
        loss = self.loss_fn(preds, yb)

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _val_step(self, X_val, y_val, X_val_extra=None):
        """Compute validation loss."""

        self.model.eval()

        with torch.no_grad():
            preds = self.model(X_val, X_val_extra)
            loss = self.loss_fn(preds, y_val)

        return loss.item()

    def _update_early_stopping(self, val_loss):
        """Update best model and check early stopping condition."""

        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_state = self.model.state_dict()  # store best weights
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return self.patience_counter >= self.patience

    def _log(
        self,
        train_loss,
        val_loss,
        epoch,
        logger=None,
        iteration=0,
        fold=None,
    ):
        """Log training metrics."""

        if logger is None:
            return

        fold_name = f"fold_{fold}"

        global_epoch = iteration * self.epochs + epoch

        logger.add_scalars(
            f"{self.model_name}/global/train_loss_updated_target",
            {
                fold_name: train_loss
            },
            global_epoch,
        )

        logger.add_scalars(
            f"{self.model_name}/global/val_loss_updated_target",
            {
                fold_name: val_loss
            },
            global_epoch,
        )

    def fit(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        X_train_extra=None,
        X_val_extra=None,
        config=None,
        logger=None,
        iteration=0,
        fold=None,
    ):
        """Train model with early stopping and return best checkpoint."""

        config = config or {}

        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32)

        X_val = np.asarray(X_val, dtype=np.float32) if X_val is not None else None
        y_val = np.asarray(y_val, dtype=np.float32) if y_val is not None else None

        if X_train.ndim == 2:
            input_dim = X_train.shape[1]
        elif X_train.ndim == 3:
            input_dim = X_train.shape[2]
        else:
            raise ValueError(
                f"Expected X_train to have 2 or 3 dimensions, got {X_train.ndim}"
            )

        self._setup(input_dim, config)

        X_train = torch.tensor(X_train).to(self.device)
        y_train = torch.tensor(y_train).to(self.device)

        if X_train_extra is not None:
            X_train_extra = torch.tensor(
                np.asarray(X_train_extra, dtype=np.float32)
            ).to(self.device)

            dataset = torch.utils.data.TensorDataset(
                X_train,
                X_train_extra,
                y_train,
            )
        else:
            dataset = torch.utils.data.TensorDataset(
                X_train,
                y_train,
            )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        if X_val is not None:
            X_val = torch.tensor(X_val).to(self.device)
            y_val = torch.tensor(y_val).to(self.device)

            if X_val_extra is not None:
                X_val_extra = torch.tensor(
                    np.asarray(X_val_extra, dtype=np.float32)
                ).to(self.device)

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.epochs):

            self.model.train()
            epoch_loss = 0.0

            for batch in loader:

                if X_train_extra is not None:
                    xb, extra, yb = batch
                    xb = xb.to(self.device)
                    extra = extra.to(self.device)
                    yb = yb.to(self.device)

                    epoch_loss += self._train_step(
                        xb,
                        yb,
                        extra,
                    ) * len(xb)

                else:
                    xb, yb = batch
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)

                    epoch_loss += self._train_step(
                        xb,
                        yb,
                    ) * len(xb)

            train_loss = epoch_loss / len(dataset)

            if X_val is not None:
                val_loss = self._val_step(
                    X_val,
                    y_val,
                    X_val_extra,
                )
            else:
                val_loss = train_loss

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            self._log(train_loss, val_loss, epoch, logger, iteration, fold)

            if self._update_early_stopping(val_loss):
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        return {
            "best_model": self,
            "best_loss": self.best_loss,
            "history": history,
        }

    def predict(self, X, extra_features=None, batch_size=4):
        """Run batched inference using trained model.

        Parameters
        ----------
        X : array-like
            Input sequences.
        extra_features : array-like, optional
            Additional features corresponding to X.
        batch_size : int, default=4
            Number of samples processed simultaneously on the GPU.

        Returns
        -------
        np.ndarray
            Predictions.
        """

        self.model.eval()

        # Keep the full dataset on CPU.
        X = torch.from_numpy(
            np.asarray(X, dtype=np.float32)
        )

        if extra_features is not None:
            extra_features = torch.from_numpy(
                np.asarray(extra_features, dtype=np.float32)
            )

        predictions = []

        with torch.inference_mode():
            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))

                # Move only the current batch to GPU.
                X_batch = X[start:end].to(self.device)

                extra_batch = None
                if extra_features is not None:
                    extra_batch = extra_features[start:end].to(self.device)

                # Forward pass.
                preds = self.model(X_batch, extra_batch)

                # Immediately move predictions back to CPU.
                predictions.append(preds.cpu())

                # Release GPU references before processing next batch.
                del X_batch
                del extra_batch
                del preds

        return torch.cat(predictions, dim=0).numpy()