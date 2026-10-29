import { app } from "../../../scripts/app.js";

const manualControls = new Set(["min_similarity", "max_drift"]);
const hiddenStates = new WeakMap();
const boundModes = new WeakSet();

function isConnected(node, name) {
    return node.inputs?.some((input) =>
        (input.name === name || input.widget?.name === name) && input.link != null,
    );
}

function updateVisibility(node) {
    const mode = node.widgets?.find((widget) => widget.name === "mode")?.value ?? "manual";
    const automatic = mode === "auto" && !isConnected(node, "mode");
    for (const widget of node.widgets ?? []) {
        if (!manualControls.has(widget.name)) continue;
        const converted = widget.type?.startsWith("converted-widget");
        const hide = automatic && !converted && !isConnected(node, widget.name);
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
    const mode = node.widgets?.find((widget) => widget.name === "mode");
    if (mode && !boundModes.has(mode)) {
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
    name: "mAI.PixelDriftFix.Visibility",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MAIPixelDriftFix") return;
        for (const hook of ["onNodeCreated", "onConfigure", "onConnectionsChange"]) {
            const original = nodeType.prototype[hook];
            nodeType.prototype[hook] = function (...args) {
                const result = original?.apply(this, args);
                if (hook === "onConfigure") {
                    const config = args[0];
                    const legacy = Array.isArray(config?.widgets_values)
                        && config.widgets_values.length < 3
                        && !config.inputs?.some((input) => input.name === "mode");
                    const mode = this.widgets?.find((widget) => widget.name === "mode");
                    if (legacy && mode) mode.value = "manual";
                }
                queueMicrotask(() => bindVisibility(this));
                return result;
            };
        }
    },
});
