import { app } from "../../../scripts/app.js";

const presetControls = new Set(["preset", "enabled", "strength", "detail_scale", "device"]);
const hiddenStates = new WeakMap();
const boundPresets = new WeakSet();
const storingNodes = new WeakSet();

function isConnected(node, name) {
    return node.inputs?.some((input) =>
        (input.name === name || input.widget?.name === name) && input.link != null,
    );
}

function movePresetFirst(node) {
    const index = node.widgets?.findIndex((widget) => widget.name === "preset") ?? -1;
    if (index > 0) node.widgets.unshift(node.widgets.splice(index, 1)[0]);
}

function updateVisibility(node) {
    if (storingNodes.has(node)) return;
    movePresetFirst(node);
    const preset = node.widgets?.find((widget) => widget.name === "preset");
    const refinement = preset?.value === "Prepare for refinement" && !isConnected(node, "preset");
    for (const widget of node.widgets ?? []) {
        if (presetControls.has(widget.name)) continue;
        // Keep converted controls and all input sockets owned by the frontend.
        const converted = widget.type?.startsWith("converted-widget");
        const hide = refinement && !converted && !isConnected(node, widget.name);
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
    const preset = node.widgets?.find((widget) => widget.name === "preset");
    if (preset && !boundPresets.has(preset)) {
        boundPresets.add(preset);
        const callback = preset.callback;
        preset.callback = function (...args) {
            const result = callback?.apply(this, args);
            updateVisibility(node);
            return result;
        };
    }
    updateVisibility(node);
}

function withStoredOrder(node, order, operation) {
    if (!node.widgets || storingNodes.has(node)) return operation();
    storingNodes.add(node);
    try {
        // LiteGraph stores widget values by position. Restore the Python schema's
        // released order only while loading/saving, then display preset first.
        node.widgets.sort((left, right) =>
            (order.get(left.name) ?? Infinity) - (order.get(right.name) ?? Infinity),
        );
        return operation();
    } finally {
        storingNodes.delete(node);
        movePresetFirst(node);
    }
}

app.registerExtension({
    name: "mAI.ImagePostEffects.Presets",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MAIImagePostEffects") return;
        const names = [
            ...Object.keys(nodeData.input?.required ?? {}),
            ...Object.keys(nodeData.input?.optional ?? {}),
        ];
        const storedOrder = new Map(names.map((name, index) => [name, index]));
        for (const hook of ["configure", "serialize"]) {
            const original = nodeType.prototype[hook];
            nodeType.prototype[hook] = function (...args) {
                return withStoredOrder(this, storedOrder, () => original?.apply(this, args));
            };
        }
        for (const hook of ["onNodeCreated", "onConfigure", "onConnectionsChange"]) {
            const original = nodeType.prototype[hook];
            nodeType.prototype[hook] = function (...args) {
                const result = original?.apply(this, args);
                // Restored values and saved sizing are available after configure.
                queueMicrotask(() => bindVisibility(this));
                return result;
            };
        }
    },
});
