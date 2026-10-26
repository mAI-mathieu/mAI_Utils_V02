import { app } from "../../../scripts/app.js";

const hiddenStates = new WeakMap();
const boundMethods = new WeakSet();

function isConnected(node, name) {
    return node.inputs?.some((input) =>
        (input.name === name || input.widget?.name === name) && input.link != null,
    );
}

function updateVisibility(node) {
    const method = node.widgets?.find((widget) => widget.name === "method")?.value ?? "color";
    // A connected method can change at execution time, so keep both settings available.
    const dynamicMethod = isConnected(node, "method");
    const visible = {
        color: dynamicMethod || method === "color",
        inpaint_radius: dynamicMethod || method === "telea" || method === "navier_stokes",
    };
    for (const widget of node.widgets ?? []) {
        if (!(widget.name in visible)) continue;
        // Keep converted widgets and sockets under ComfyUI's control.
        const converted = widget.type?.startsWith("converted-widget");
        const hide = !visible[widget.name] && !converted && !isConnected(node, widget.name);
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
    const method = node.widgets?.find((widget) => widget.name === "method");
    if (method && !boundMethods.has(method)) {
        boundMethods.add(method);
        const callback = method.callback;
        method.callback = function (...args) {
            const result = callback?.apply(this, args);
            updateVisibility(node);
            return result;
        };
    }
    updateVisibility(node);
}

app.registerExtension({
    name: "mAI.MaskFill.Visibility",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MAIMaskFill") return;
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
