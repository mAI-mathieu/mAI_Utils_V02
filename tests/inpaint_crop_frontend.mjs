import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

let extension;
globalThis.testApp = { registerExtension(value) { extension = value; } };
const source = await readFile(new URL("../web/js/inpaint_crop_separate_stitch_mask.js", import.meta.url), "utf8");
await import(`data:text/javascript;base64,${Buffer.from(source.replace(
    'import { app } from "../../../scripts/app.js";', "const app = globalThis.testApp;",
)).toString("base64")}`);
delete globalThis.testApp;

class Node {
    constructor() {
        this.size = [400, 1000];
        this.inputs = [{ name: "image", link: 1 }, { name: "mask", link: 2 }];
        this.outputs = [{ name: "stitcher", links: [3] }, { name: "cropped_image", links: [4] }];
        const values = {
            preresize: false, preresize_mode: "ensure minimum resolution",
            preresize_min_width: 1024, preresize_min_height: 1024,
            preresize_max_width: 2048, preresize_max_height: 2048,
            extend_for_outpainting: false, extend_up_factor: 1.5,
            extend_down_factor: 1, extend_left_factor: 1, extend_right_factor: 1,
            context_from_mask_extend_factor: 1.2, output_resize_to_target_size: false,
            output_target_width: 512, output_target_height: 512, output_padding: "32",
            stitch_mask_mode: "rectangle (full crop)", stitch_mask_expand_pixels: 32,
            stitch_mask_blend_pixels: 32, use_mask_max_size: false,
            max_crop_width: 1024, max_crop_height: 1024, crop_multiple_of: 64,
        };
        this.widgets = Object.entries(values).map(([name, value]) => ({
            name, value, type: typeof value === "boolean" ? "toggle" : "number",
            callback() { return "callback"; }, serializeValue() { return this.value; },
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
extension.beforeRegisterNodeDef(Node, { name: "MAIInpaintCropSeparateStitchMask" });
const node = new Node();
const widgets = [...node.widgets];
const slots = JSON.stringify([node.inputs, node.outputs]);
const values = node.widgets.map((w) => w.value);
const widget = (name) => node.widgets.find((w) => w.name === name);
const change = (name, value) => {
    widget(name).value = value;
    assert.equal(widget(name).callback(value), "callback");
};
assert.equal(node.onNodeCreated(), "created");
await Promise.resolve();
for (const name of ["preresize_mode", "preresize_min_width", "preresize_max_height",
    "extend_up_factor", "output_target_width", "stitch_mask_expand_pixels", "max_crop_width"]) {
    assert.equal(widget(name).hidden, true, name);
}
assert.deepEqual(node.widgets.map((w) => w.value), values);
change("preresize", true);
assert.ok(!widget("preresize_min_width").hidden);
assert.equal(widget("preresize_max_width").hidden, true);
change("preresize_mode", "ensure maximum resolution");
assert.equal(widget("preresize_min_width").hidden, true);
assert.ok(!widget("preresize_max_width").hidden);
change("preresize_mode", "ensure minimum and maximum resolution");
assert.ok(!widget("preresize_min_width").hidden && !widget("preresize_max_width").hidden);
change("extend_for_outpainting", true);
assert.ok(!widget("extend_up_factor").hidden);
change("output_resize_to_target_size", true);
assert.ok(!widget("output_target_width").hidden);
change("stitch_mask_mode", "extended mask");
assert.ok(!widget("stitch_mask_expand_pixels").hidden);
const expandedHeight = node.size[1];
change("use_mask_max_size", true);
assert.ok(node.size[1] < expandedHeight);
for (const name of ["preresize", "preresize_mode", "extend_for_outpainting", "extend_up_factor",
    "output_resize_to_target_size", "output_target_width", "output_padding", "context_from_mask_extend_factor"]) {
    assert.equal(widget(name).hidden, true, name);
}
assert.ok(!widget("max_crop_width").hidden && !widget("crop_multiple_of").hidden);
const saved = node.widgets.map((w) => w.value);
assert.equal(node.onConfigure(saved), "configured");
await Promise.resolve();
assert.equal(widget("preresize").hidden, true);
assert.deepEqual(node.widgets.map((w) => w.serializeValue()), saved);
change("use_mask_max_size", false);
assert.ok(!widget("extend_up_factor").hidden);
assert.equal(widget("extend_up_factor").value, 1.5);
assert.equal(widget("max_crop_width").hidden, true);

node.inputs.push({ name: "max_crop_width", widget: { name: "max_crop_width" }, link: 42 });
node.onConnectionsChange();
await Promise.resolve();
assert.ok(!widget("max_crop_width").hidden);
node.inputs.pop();
node.onConnectionsChange();
await Promise.resolve();
assert.equal(widget("max_crop_width").hidden, true);
node.inputs.push({ name: "use_mask_max_size", link: 43 });
node.onConnectionsChange();
await Promise.resolve();
assert.ok(!widget("max_crop_width").hidden && !widget("preresize").hidden);
node.inputs.pop();
change("use_mask_max_size", false);
assert.deepEqual(node.widgets, widgets);
assert.equal(JSON.stringify([node.inputs, node.outputs]), slots);
assert.ok(node.widgets.every((w) => w.type !== "mai_hidden"));
assert.equal(node.size[0], 400);
console.log("Crop visibility, booleans, modes, saved values and sockets passed.");
