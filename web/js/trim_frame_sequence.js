import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "mAI.TrimFrameSequence.SavedSettings",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MAITrimFrameSequence") return;
        const original = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (config, ...args) {
            const result = original?.call(this, config, ...args);
            const values = config?.widgets_values;
            if (!Array.isArray(values)) return result;
            const [mode, amount] = values;
            // Old workflows stored a dropdown followed by one shared count.
            // New workflows already store two numbers and need no conversion.
            if (!["start", "end", "both ends"].includes(mode)) return result;
            const start = this.widgets?.find((widget) => widget.name === "trim_start");
            const end = this.widgets?.find((widget) => widget.name === "trim_end");
            if (!start || !end) return result;
            start.value = mode === "end" ? 0 : amount;
            end.value = mode === "start" ? 0 : amount;
            return result;
        };
    },
});
