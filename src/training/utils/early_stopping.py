class EarlyStopping:
    """
    Implementa Early Stopping basado en una métrica.

    Parameters
    ----------
    patience
        Número máximo de épocas sin mejora.

    min_delta
        Mejora mínima requerida para considerar una nueva mejor métrica.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:

        self.patience = patience

        self.min_delta = min_delta

        self.counter = 0

        if mode == "min":
            self.best_score = float("inf")
        else:
            self.best_score = float("-inf")

        self.mode = mode

        self.stop = False

    def __call__(
        self,
        score: float,
    ) -> bool:
        """
        Evalúa si debe detener el entrenamiento.

        Parameters
        ----------
        score
            Valor de la métrica (por ejemplo Validation Loss).

        Returns
        -------
        bool
            True si debe detenerse el entrenamiento.
        """

        # IMPROVEMENT
        if self.mode == "min":
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
                return False
        else:
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
                return False

        # NO IMPROVEMENT
        self.counter += 1

        if self.counter >= self.patience:

            self.stop = True

        return self.stop