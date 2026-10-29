import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

let extension;
globalThis.testApp = { registerExtension(value) { extension = value; } };
const source = await readFile(new URL("../web/js/image_preserve_source.js", import.meta.url), "utf8");
await import(`data:text/javascript;base64,${Buffer.from(source.replace(
    'import { app } from "../../../scripts/app.js";', "const app = globalThis.testApp;",
)).toString("base64")}`);
delete globalThis.testApp;

class Node {
    constructor() {
        this.widgets = [
            { name: "difference_threshold", value: 0.08, hidden: false, type: "number" },
            { name: "comparison_blur", value: 2, type: "number" },
            { name: "blend_mode", value: "multiband", type: "combo" },
            { name: "mode", value: "manual", type: "combo", callback() { return "changed"; } },
        ];
        this.inputs = [{ name: "original_image", link: 1 }, { name: "edited_image", link: 2 }];
        this.outputs = [{ name: "image", links: [3] }, { name: "change_mask", links: [4] }];
        this.size = [300, 400];
    }
    onConfigure(config) {
        config.widgets_values?.forEach((value, index) => { this.widgets[index].value = value; });
        return "configured";
    }
    computeSize() { return [300, this.widgets.filter((widget) => !widget.hidden).length * 30]; }
    setSize(size) { this.size = size; }
    setDirtyCanvas() {}
}

const originalConfigure = Node.prototype.onConfigure;
extension.beforeRegisterNodeDef(Node, { name: "OtherNode" });
assert.equal(Node.prototype.onConfigure, originalConfigure);
extension.beforeRegisterNodeDef(Node, { name: "MAIImagePreserveSource" });

const node = new Node();
const sockets = JSON.stringify([node.inputs, node.outputs]);
node.onNodeCreated();
await Promise.resolve();
const mode = node.widgets[3];
assert.equal(mode.callback(), "changed");
assert.equal(node.widgets[0].hidden, false);
mode.value = "auto";
mode.callback();
assert.ok(node.widgets.slice(0, 3).every((widget) => widget.hidden));
assert.equal(mode.hidden, undefined);
assert.equal(JSON.stringify([node.inputs, node.outputs]), sockets);

// Manual values and their serialized positions survive toggling and reloading.
node.widgets[0].value = 0.123;
const saved = node.widgets.map((widget) => widget.value);
const restored = new Node();
assert.equal(restored.onConfigure({ widgets_values: saved }), "configured");
await Promise.resolve();
assert.equal(restored.widgets[0].hidden, true);
assert.deepEqual(restored.widgets.map((widget) => widget.value), saved);
restored.widgets[3].value = "manual";
restored.widgets[3].callback();
assert.equal(restored.widgets[0].hidden, false);
assert.equal(restored.widgets[1].hidden, undefined);
assert.equal(restored.widgets[0].value, 0.123);

// Legacy widgets_values has no mode and keeps the original manual behavior.
const legacy = new Node();
legacy.onConfigure({ widgets_values: [0.12, 4, "feather"] });
await Promise.resolve();
assert.equal(legacy.widgets[3].value, "manual");
assert.equal(legacy.widgets[0].hidden, false);

// Connected mode is determined at execution; expose every tuning control.
node.inputs.push({ name: "mode", link: 10 });
node.onConnectionsChange();
await Promise.resolve();
assert.equal(node.widgets[0].hidden, false);
node.inputs.pop();
node.inputs.push({ name: "comparison_blur", link: 11 });
node.widgets[1].type = "converted-widget";
node.onConnectionsChange();
await Promise.resolve();
assert.equal(node.widgets[0].hidden, true);
assert.equal(node.widgets[1].hidden, undefined);
assert.equal(node.inputs.at(-1).link, 11);
console.log("Mode visibility, saved values, legacy loading and connected sockets passed.");
