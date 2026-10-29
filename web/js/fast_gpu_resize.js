import { app } from "../../../scripts/app.js";

const pixelPreservingModes = new Set([
    "pad_to_multiple", "pad_to_multiple_edge", "center_crop",
]);
const samplingControls = new Set(["method", "antialias", "precision"]);
const hiddenStates = new WeakMap();
const boundModes = new WeakSet();

function updateVisibility(node) {
    const mode = node.widgets?.find((widget) => widget.name === "resize_mode");
    if (!mode) return;
    for (const widget of node.widgets) {
        if (!samplingControls.has(widget.name) && widget.name !== "fill_color") continue;
        // Converted widgets belong to the frontend; leave their sockets intact.
        const converted = widget.type?.startsWith("converted-widget");
        const hide = !converted && (widget.name === "fill_color"
            ? mode.value !== "pad_to_multiple"
            : pixelPreservingModes.has(mode.value));
        if (hide && !hiddenStates.has(widget)) {
            hiddenStates.set(widget, widget.hidden);
            widget.hidden = true;
        } else if (!hide && hiddenStates.has(widget)) {
            widget.hidden = hiddenStates.get(widget);
            hiddenStates.delete(widget);
        }
    }
    const size = node.computeSize();
    node.setSize([Math.max(node.size[0], size[0]), size[1]]);
    node.setDirtyCanvas(true, true);
}

function bindVisibility(node) {
    const mode = node.widgets?.find((widget) => widget.name === "resize_mode");
    if (!mode) return;
    if (!boundModes.has(mode)) {
        boundModes.add(mode);
        const callback = mode.callback;
        mode.callback = function (...args) {
            const result = callback?.apply(this, args);
            updateVisibility(node);
            return result;
        };
    }
    updateVisibility(node);
}

app.registerExtension({
    name: "mAI.FastGPUResize.Visibility",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MAIFastGPUResize") return;
        for (const hook of ["onNodeCreated", "onConfigure", "onConnectionsChange"]) {
            const original = nodeType.prototype[hook];
            nodeType.prototype[hook] = function (...args) {
                const result = original?.apply(this, args);
                queueMicrotask(() => bindVisibility(this));
                return result;
            };
        }
    },
});
