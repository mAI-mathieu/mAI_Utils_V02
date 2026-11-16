import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "mAI.ImageBlend.SavedSettings",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MAIImageBlend") return;

        const configure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (config, ...args) {
            const result = configure?.call(this, config, ...args);
            const legacy = this.widgets?.find((widget) => widget.name === "blend_list");
            if (!legacy) return result;

            const values = config?.widgets_values;
            const legacyLinked = config?.inputs?.some((input) =>
                input.name === "blend_list" && input.link != null,
            );
            if (Array.isArray(values) && values.length === 1 && (typeof values[0] === "string" || legacyLinked)) {
                const text = legacyLinked ? "" : values[0];
                const entries = text.split(/[,\r\n]/).map((entry) => entry.trim()).filter(Boolean);
                const scalar = entries.length === 1 ? entries[0].replace(/%$/, "").trim() : "";
                const number = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(scalar)
                    ? Number(scalar)
                    : NaN;
                const migrate = number >= 0 && number <= 100;
                const mode = this.widgets.find((widget) => widget.name === "blend_mode");
                const percentage = this.widgets.find((widget) => widget.name === "blend_percentage");
                if (mode) mode.value = "normal";
                if (percentage) percentage.value = migrate ? number : 50;
                legacy.value = migrate ? "" : text;
            }
            legacy.hidden = !(legacy.value || legacyLinked);
            return result;
        };
    },
});
