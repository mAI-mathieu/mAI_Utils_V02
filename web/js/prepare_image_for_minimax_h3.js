import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "mAI.PrepareImageForMinimaxH3.SavedSettings",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MAIPrepareImageForMinimaxH3") return;
        const original = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = original?.apply(this, args);
            const megapixels = this.widgets?.find((widget) => widget.name === "target_megapixels");
            if (typeof megapixels?.value === "string") {
                const value = Number(megapixels.value.trim().replace(/\s*MP$/i, ""));
                if (Number.isFinite(value) && value >= 0.2 && value <= 2.0) {
                    megapixels.value = value;
                }
            }
            const mode = this.widgets?.find((widget) => widget.name === "resize_mode");
            if (mode?.value === "Short side 768 px") {
                mode.value = "Short side";
            }
            return result;
        };
    },
});
