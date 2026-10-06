import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

let extension;
globalThis.testApp = { registerExtension(value) { extension = value; } };
const source = await readFile(new URL("../web/js/trim_frame_sequence.js", import.meta.url), "utf8");
await import(`data:text/javascript;base64,${Buffer.from(source.replace(
    'import { app } from "../../../scripts/app.js";',
    "const app = globalThis.testApp;",
)).toString("base64")}`);
delete globalThis.testApp;

class Node {
    constructor() {
        this.widgets = [{ name: "trim_start", value: 0 }, { name: "trim_end", value: 0 }];
        this.inputs = [{ name: "frames", link: 10 }];
        this.outputs = [{ name: "frames", links: [11] }, { name: "frame_count", links: [12] }];
    }
    onConfigure(config) {
        config.widgets_values?.forEach((value, index) => { this.widgets[index].value = value; });
        return "configured";
    }
}

const original = Node.prototype.onConfigure;
extension.beforeRegisterNodeDef(Node, { name: "OtherNode" });
assert.equal(Node.prototype.onConfigure, original);
extension.beforeRegisterNodeDef(Node, { name: "MAITrimFrameSequence" });

for (const [mode, amount, expected] of [
    ["start", 2, [2, 0]], ["end", 3, [0, 3]], ["both ends", 4, [4, 4]],
    ["start", 0, [0, 0]], ["end", 0, [0, 0]], ["both ends", 0, [0, 0]],
]) {
    const node = new Node();
    const slots = JSON.stringify([node.inputs, node.outputs]);
    const config = { widgets_values: [mode, amount] };
    assert.equal(node.onConfigure(config), "configured");
    assert.deepEqual(node.widgets.map((widget) => widget.value), expected);
    assert.deepEqual(config.widgets_values, [mode, amount]);
    assert.equal(JSON.stringify([node.inputs, node.outputs]), slots);
    // Saving and loading the new widget values must not migrate them again.
    node.onConfigure({ widgets_values: expected });
    assert.deepEqual(node.widgets.map((widget) => widget.value), expected);
}

const node = new Node();
for (const values of [[2, 3], [0, 5], [4, 0], [0, 0]]) {
    node.onConfigure({ widgets_values: values });
    assert.deepEqual(node.widgets.map((widget) => widget.value), values);
}
assert.equal(node.onConfigure({}), "configured");
// Invalid legacy counts stay invalid so backend validation can report them.
node.onConfigure({ widgets_values: ["both ends", -1] });
assert.deepEqual(node.widgets.map((widget) => widget.value), [-1, -1]);
console.log("Legacy saved settings, new counts, hooks and sockets passed.");
