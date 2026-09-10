class MAITypeConverterNode:
    CATEGORY = "mAI / Utils"
    RETURN_TYPES = ("BOOLEAN", "STRING", "INT", "FLOAT")
    RETURN_NAMES = ("boolean", "string", "int", "float")
    FUNCTION = "convert"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_type": (["string", "int", "float", "boolean"], {"default": "string"}),
                "boolean": ("BOOLEAN", {"default": False}),
                "string": ("STRING", {"default": ""}),
                "int": ("INT", {"default": 0}),
                "float": ("FLOAT", {"default": 0.0}),
                "strict": ("BOOLEAN", {"default": False}),
            }
        }

    def convert(self, source_type, boolean, string, int, float, strict):
        value = self._select_source(source_type, boolean, string, int, float)
        return (
            self._to_bool(value, strict),
            self._to_string(value),
            self._to_int(value, strict),
            self._to_float(value, strict),
        )

    def _select_source(self, source_type, boolean, string, int, float):
        if source_type == "boolean":
            return boolean
        if source_type == "string":
            return string
        if source_type == "int":
            return int
        if source_type == "float":
            return float

        raise ValueError(f"Unknown source_type: {source_type}")

    def _to_string(self, value):
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def _to_bool(self, value, strict):
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value != 0
        if isinstance(value, float):
            return value != 0.0
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"true", "1", "yes", "y", "on"}:
                return True
            if text in {"false", "0", "no", "n", "off", ""}:
                return False
            try:
                return float(text) != 0.0
            except ValueError:
                if strict:
                    raise ValueError(f"Cannot convert string to boolean: {value}")
                return False

        if strict:
            raise ValueError(f"Cannot convert value to boolean: {value}")
        return False

    def _to_int(self, value, strict):
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            text = value.strip()
            try:
                return int(float(text))
            except ValueError:
                if strict:
                    raise ValueError(f"Cannot convert string to int: {value}")
                return 0

        if strict:
            raise ValueError(f"Cannot convert value to int: {value}")
        return 0

    def _to_float(self, value, strict):
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        if isinstance(value, int):
            return float(value)
        if isinstance(value, float):
            return value
        if isinstance(value, str):
            text = value.strip()
            try:
                return float(text)
            except ValueError:
                if strict:
                    raise ValueError(f"Cannot convert string to float: {value}")
                return 0.0

        if strict:
            raise ValueError(f"Cannot convert value to float: {value}")
        return 0.0
