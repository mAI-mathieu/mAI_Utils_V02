import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "mAI.PrepareImageForMinimaxH3.SavedSettings",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MAIPrepareImageForMinimaxH3") return;
        const original = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = original?.apply(this, args);
            const mode = this.widgets?.find((widget) => widget.name === "resize_mode");
            if (mode?.value === "Short side 768 px") {
                mode.value = "Short side";
            }
            return result;
        };
    },
});
