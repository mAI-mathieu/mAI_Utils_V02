// Test the real extension without requiring a running ComfyUI frontend.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

let extension;
globalThis.testApp = { registerExtension(value) { extension = value; } };
const source = await readFile(new URL("../web/js/fast_gpu_resize.js", import.meta.url), "utf8");
await import(`data:text/javascript;base64,${Buffer.from(source.replace(
    'import { app } from "../../../scripts/app.js";', "const app = globalThis.testApp;",
)).toString("base64")}`);
delete globalThis.testApp;

class Node {
    constructor() {
        this.size = [420, 1000];
        this.inputs = [{ name: "image", link: 1 }, { name: "mask", link: 2 }];
        this.outputs = ["image", "width", "height", "method_used", "mask", "stitcher"];
        this.widgets = Object.entries({
            width: 1280, height: 720, resize_mode: "exact", method: "lanczos4",
            antialias: true, multiple_of: 8, device: "auto", precision: "fp16", chunk_size: 0,
            fill_color: "#12ABEF",
        }).map(([name, value]) => ({
            name, value, type: "combo", callback() { return "callback"; },
            serializeValue() { return this.value; },
        }));
    }
    computeSize() { return [300, 80 + this.widgets.filter((w) => !w.hidden).length * 24]; }
    setSize(size) { this.size = size; }
    setDirtyCanvas() {}
    onNodeCreated() { return "created"; }
    onConfigure(values) {
        this.widgets.forEach((widget, index) => { widget.value = values[index]; });
        return "configured";
    }
}

const original = Node.prototype.onNodeCreated;
extension.beforeRegisterNodeDef(Node, { name: "OtherNode" });
assert.equal(Node.prototype.onNodeCreated, original);
extension.beforeRegisterNodeDef(Node, { name: "MAIFastGPUResize" });
const node = new Node();
const widgets = [...node.widgets];
const slots = JSON.stringify([node.inputs, node.outputs]);
const widget = (name) => node.widgets.find((w) => w.name === name);
const mode = widget("resize_mode");
assert.equal(node.onNodeCreated(), "created");
await Promise.resolve();
const expandedHeight = node.size[1];
assert.equal(widget("fill_color").hidden, true);

for (const value of ["pad_to_multiple", "pad_to_multiple_edge", "center_crop"]) {
    mode.value = value;
    assert.equal(mode.callback(value), "callback");
    for (const name of ["method", "antialias", "precision"]) {
        assert.equal(widget(name).hidden, true);
    }
    assert.equal(widget("fill_color").hidden, value === "pad_to_multiple" ? undefined : true);
    assert.ok(node.size[1] < expandedHeight);
    assert.equal(node.size[0], 420);
    const saved = node.widgets.map((w) => w.serializeValue());
    assert.equal(node.onConfigure(saved), "configured");
    await Promise.resolve();
    assert.deepEqual(node.widgets.map((w) => w.serializeValue()), saved);
    assert.equal(widget("method").hidden, true);
    assert.equal(widget("fill_color").hidden, value === "pad_to_multiple" ? undefined : true);
}

for (const value of ["exact", "keep_aspect_fit", "keep_aspect_fill", "keep_aspect"]) {
    mode.value = value;
    mode.callback(value);
    for (const name of ["method", "antialias", "precision"]) {
        assert.equal(widget(name).hidden, undefined);
    }
    assert.equal(node.size[1], expandedHeight);
    assert.equal(widget("fill_color").hidden, true);
}
assert.equal(widget("fill_color").value, "#12ABEF");
assert.equal(widget("method").value, "lanczos4");
assert.equal(widget("antialias").value, true);
assert.equal(widget("precision").value, "fp16");
assert.deepEqual(node.widgets, widgets);
assert.equal(JSON.stringify([node.inputs, node.outputs]), slots);

// Connected controls retain sockets and serialization, even when ignored.
widget("method").type = "converted-widget";
widget("method").hidden = true;
node.inputs.push({ name: "method", widget: { name: "method" }, link: 42 });
mode.value = "center_crop";
mode.callback(mode.value);
node.onConnectionsChange();
await Promise.resolve();
mode.value = "exact";
mode.callback(mode.value);
assert.equal(widget("method").hidden, true);
assert.equal(widget("method").type, "converted-widget");
assert.equal(node.inputs.at(-1).link, 42);

// A converted fill color keeps its socket in every mode as well.
widget("fill_color").type = "converted-widget";
widget("fill_color").hidden = true;
node.inputs.push({ name: "fill_color", widget: { name: "fill_color" }, link: 43 });
for (const value of ["pad_to_multiple", "exact"]) {
    mode.value = value;
    mode.callback(value);
    assert.equal(widget("fill_color").type, "converted-widget");
    assert.equal(node.inputs.at(-1).link, 43);
}
console.log("Resize mode visibility, saved values, lifecycle and socket preservation passed.");
