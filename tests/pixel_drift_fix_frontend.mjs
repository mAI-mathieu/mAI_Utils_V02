import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

let extension;
globalThis.testApp = { registerExtension(value) { extension = value; } };
const source = await readFile(new URL("../web/js/pixel_drift_fix.js", import.meta.url), "utf8");
await import(`data:text/javascript;base64,${Buffer.from(source.replace(
    'import { app } from "../../../scripts/app.js";', "const app = globalThis.testApp;",
)).toString("base64")}`);
delete globalThis.testApp;

class Node {
    constructor() {
        this.widgets = [
            { name: "min_similarity", value: 0.6, hidden: false, type: "number" },
            { name: "max_drift", value: 0.15, type: "number" },
            { name: "mode", value: "auto", type: "combo", callback() { return "changed"; } },
        ];
        this.inputs = [{ name: "source_image", link: 1 }, { name: "edited_image", link: 2 }];
        this.outputs = [{ name: "image", links: [3] }, { name: "alignment_info", links: [4] }];
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
extension.beforeRegisterNodeDef(Node, { name: "MAIPixelDriftFix" });
const node = new Node();
const sockets = JSON.stringify([node.inputs, node.outputs]);
node.onNodeCreated();
await Promise.resolve();
assert.equal(node.widgets[2].callback(), "changed");
assert.ok(node.widgets.slice(0, 2).every((widget) => widget.hidden));
assert.equal(JSON.stringify([node.inputs, node.outputs]), sockets);

// Manual values retain their positions and survive toggles and reloads.
node.widgets[2].value = "manual";
node.widgets[2].callback();
assert.equal(node.widgets[0].hidden, false);
assert.equal(node.widgets[1].hidden, undefined);
node.widgets[0].value = 0.75;
node.widgets[2].value = "auto";
node.widgets[2].callback();
const saved = node.widgets.map((widget) => widget.value);
const restored = new Node();
assert.equal(restored.onConfigure({ widgets_values: saved }), "configured");
await Promise.resolve();
assert.equal(restored.widgets[0].hidden, true);
assert.deepEqual(restored.widgets.map((widget) => widget.value), saved);

// Workflows saved before mode existed continue to use their manual settings.
const legacy = new Node();
legacy.onConfigure({ widgets_values: [0.8, 0.12] });
await Promise.resolve();
assert.equal(legacy.widgets[2].value, "manual");
assert.equal(legacy.widgets[0].hidden, false);
assert.deepEqual(legacy.widgets.map((widget) => widget.value), [0.8, 0.12, "manual"]);

// Connected mode/converted controls stay visible and their sockets stay intact.
node.inputs.push({ name: "mode", link: 10 });
node.onConnectionsChange();
await Promise.resolve();
assert.equal(node.widgets[0].hidden, false);
node.inputs.pop();
node.inputs.push({ name: "max_drift", link: 11 });
node.widgets[1].type = "converted-widget";
node.onConnectionsChange();
await Promise.resolve();
assert.equal(node.widgets[0].hidden, true);
assert.equal(node.widgets[1].hidden, undefined);
assert.equal(node.inputs.at(-1).link, 11);
console.log("Auto defaults, visibility, legacy loading and connected sockets passed.");
