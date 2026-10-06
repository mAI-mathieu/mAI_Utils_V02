import { app } from "../../../scripts/app.js";

// Widgets stay serialized, and sockets stay fixed. Python owns all defaults.
app.registerExtension({
    name: "mAI.AutoSeamlessLoop.AdvancedControls",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MAIAutoSeamlessLoop") return;
        const advancedNames = new Set(Object.keys(nodeData.input?.optional ?? {}));
        const update = function () {
            const visible = this.widgets?.find(w => w.name === "advanced_controls")?.value;
            for (const widget of this.widgets ?? []) {
                if (!advancedNames.has(widget.name)) continue;
                if (!widget.maiOriginal) {
                    widget.maiOriginal = { type: widget.type, computeSize: widget.computeSize, draw: widget.draw };
                }
                // A connected optional input must remain accessible.
                const connected = this.inputs?.some(input => input.name === widget.name && input.link != null);
                widget.type = visible || connected ? widget.maiOriginal.type : "mai_hidden";
                widget.computeSize = visible || connected ? widget.maiOriginal.computeSize : () => [0, -4];
                widget.draw = visible || connected ? widget.maiOriginal.draw : () => {};
            }
            this.setSize(this.computeSize());
            this.graph?.setDirtyCanvas(true, true);
        };
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            const toggle = this.widgets?.find(w => w.name === "advanced_controls");
            if (toggle) {
                const callback = toggle.callback;
                toggle.callback = (...values) => {
                    callback?.apply(toggle, values);
                    update.call(this);
                };
            }
            update.call(this);
            return result;
        };
        for (const hook of ["onConfigure", "onConnectionsChange"]) {
            const original = nodeType.prototype[hook];
            nodeType.prototype[hook] = function (...args) {
                const result = original?.apply(this, args);
                update.call(this);
                return result;
            };
        }
    },
});
