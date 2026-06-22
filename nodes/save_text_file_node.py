try:
    from ..utils.text_file_saver import save_text_file
except ImportError:
    from utils.text_file_saver import save_text_file


class MAISaveTextFile:
    CATEGORY = "mAI / IO"
    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("file_path", "saved")
    FUNCTION = "save"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "file_name": ("STRING", {"default": "output.txt"}),
                "folder": ("STRING", {"default": ""}),
                "overwrite": ("BOOLEAN", {"default": True}),
            }
        }

    def save(self, text, file_name, folder, overwrite):
        file_path = save_text_file(text, file_name, folder, overwrite=overwrite)
        return (file_path, True)
