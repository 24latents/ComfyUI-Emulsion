try:
    from .emulsion import EmulsionGrade
    from .extract import EmulsionExtract
except ImportError:  # import outside the package (pytest / direct tests)
    from emulsion import EmulsionGrade
    from extract import EmulsionExtract

NODE_CLASS_MAPPINGS = {
    "EmulsionGrade": EmulsionGrade,
    "EmulsionExtract": EmulsionExtract,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EmulsionGrade": "Emulsion Grade",
    "EmulsionExtract": "Emulsion Extract",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
