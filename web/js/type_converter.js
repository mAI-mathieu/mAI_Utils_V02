import { app } from "../../../scripts/app.js";

function updateOutput(node, disconnect = false) {
    const widget = node.widgets?.find((item) => item.name === "output_type");
    const output = node.outputs?.[0];
    if (!widget || !output) return;

    const type = String(widget.value).toUpperCase();
    if (!["STRING", "INT", "FLOAT", "BOOLEAN"].includes(type)) return;

    if (disconnect && output.type !== type && output.links?.length) {
        // Existing consumers may require the previous type. Reconnect after changing it.
        node.disconnectOutput(0);
    }
    output.type = type;
    output.name = widget.value;
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "mAI.TypeConverter",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MAITypeConverterNode") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = onNodeCreated?.apply(this, args);
            const widget = this.widgets?.find((item) => item.name === "output_type");
            if (widget) {
                const callback = widget.callback;
                widget.callback = (...callbackArgs) => {
                    const result = callback?.apply(widget, callbackArgs);
                    updateOutput(this, true);
                    return result;
                };
            }
            updateOutput(this);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = onConfigure?.apply(this, args);
            updateOutput(this);
            return result;
        };
    },
});
