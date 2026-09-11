import { app } from "../../../scripts/app.js";

const basicControls = new Set([
    "enabled", "preset", "advanced_mode", "strength", "contrast",
    "highlight_rolloff", "saturation", "color_density", "halation_strength",
    "bloom_strength", "grain_strength", "vignette_strength",
]);
const hiddenStates = new WeakMap();
const boundToggles = new WeakSet();

function updateVisibility(node) {
    const advanced = node.widgets?.find((widget) => widget.name === "advanced_mode");
    if (!advanced) return;

    for (const widget of node.widgets) {
        if (basicControls.has(widget.name)) continue;
        // Leave converted/connected controls and their sockets accessible.
        const connected = node.inputs?.some(
            (input) => input.widget?.name === widget.name && input.link != null,
        );
        const converted = widget.type?.startsWith("converted-widget");
        const hide = !advanced.value && !connected && !converted;
        if (hide && !hiddenStates.has(widget)) {
            hiddenStates.set(widget, widget.hidden);
            widget.hidden = true;
        } else if (!hide && hiddenStates.has(widget)) {
            widget.hidden = hiddenStates.get(widget);
            hiddenStates.delete(widget);
        }
    }

    // Use the frontend's native hidden flag: values, widget types, serialization
    // and socket order stay intact. Recompute height so basic mode is compact.
    const size = node.computeSize();
    node.setSize([Math.max(node.size[0], size[0]), size[1]]);
    node.setDirtyCanvas(true, true);
}

function bindVisibility(node) {
    const toggle = node.widgets?.find((widget) => widget.name === "advanced_mode");
    if (!toggle) return;
    if (!boundToggles.has(toggle)) {
        boundToggles.add(toggle);
        const callback = toggle.callback;
        toggle.callback = function (...args) {
            const result = callback?.apply(this, args);
            updateVisibility(node);
            return result;
        };
    }
    updateVisibility(node);
}

app.registerExtension({
    name: "mAI.CinematicPost.Visibility",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "mAI_CinematicPost") return;
        for (const hook of ["onNodeCreated", "onConfigure", "onConnectionsChange"]) {
            const original = nodeType.prototype[hook];
            nodeType.prototype[hook] = function (...args) {
                const result = original?.apply(this, args);
                // Wait for widget creation, restored values and saved sizing.
                queueMicrotask(() => bindVisibility(this));
                return result;
            };
        }
    },
});
