if __package__ == "nodes":
    from utils.seeded_noise import MAISeededNoise
else:
    from ..utils.seeded_noise import MAISeededNoise


_UNCONNECTED = object()
_SOURCE_TYPES = (str, int, float, bool)


class MAITypeConverterNode:
    CATEGORY = "mAI / Utils"
    RETURN_TYPES = ("BOOLEAN", "STRING", "INT", "FLOAT", "NOISE")
    RETURN_NAMES = ("boolean", "string", "int", "float", "noise")
    FUNCTION = "convert"
    DESCRIPTION = (
        "Connect a string, integer, float or boolean to value for automatic "
        "source detection. Without value, use source_type and the manual fields. "
        "The noise output uses the converted integer as a seed for "
        "SamplerCustomAdvanced."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_type": (["auto", "string", "int", "float", "boolean"], {
                    "default": "auto",
                    "tooltip": "Fallback source when value is disconnected. Auto uses the string field.",
                }),
                "boolean": ("BOOLEAN", {"default": False}),
                "string": ("STRING", {"default": ""}),
                "int": ("INT", {"default": 0, "min": -0x8000000000000000, "max": 0xffffffffffffffff}),
                "float": ("FLOAT", {"default": 0.0}),
                "strict": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "value": ("*", {
                    "tooltip": "Automatically detects STRING, INT, FLOAT or BOOLEAN; overrides the manual source.",
                }),
            },
        }

    def convert(self, source_type="auto", boolean=False, string="", int=0,
                float=0.0, strict=False, value=_UNCONNECTED):
        value = self._select_source(source_type, boolean, string, int, float, value)
        integer = self._to_int(value, strict)
        return (
            self._to_bool(value, strict),
            self._to_string(value),
            integer,
            self._to_float(value, strict),
            MAISeededNoise(integer),
        )

    def _select_source(self, source_type, boolean, string, int, float, value=_UNCONNECTED):
        if value is not _UNCONNECTED:
            if not isinstance(value, _SOURCE_TYPES):
                raise ValueError(
                    "value must be a STRING, INT, FLOAT or BOOLEAN; "
                    f"received {type(value).__name__}."
                )
            return value
        if source_type == "boolean":
            return boolean
        if source_type in {"auto", "string"}:
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
            try:
                return int(value)
            except (ValueError, OverflowError):
                if strict:
                    raise ValueError(f"Cannot convert float to int: {value}") from None
                return 0
        if isinstance(value, str):
            text = value.strip()
            try:
                # Preserve all 64 seed bits before trying decimal/scientific text.
                return int(text)
            except ValueError:
                try:
                    return int(float(text))
                except (ValueError, OverflowError):
                    if strict:
                        raise ValueError(f"Cannot convert string to int: {value}") from None
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
