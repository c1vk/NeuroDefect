class NeuroDefectError(Exception):
    """User-facing application error. The UI should display str(error)."""


class DataError(NeuroDefectError):
    pass


class TrainingError(NeuroDefectError):
    pass


class PredictionError(NeuroDefectError):
    pass
