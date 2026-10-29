// Exercise the real extension's ordering and positional serialization contract.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

let extension;
globalThis.testApp = { registerExtension(value) { extension = value; } };
const source = await readFile(new URL("../web/js/image_post_effects.js", import.meta.url), "utf8");
await import(`data:text/javascript;base64,${Buffer.from(source.replace(
    'import { app } from "../../../scripts/app.js";', "const app = globalThis.testApp;",
)).toString("base64")}`);
delete globalThis.testApp;

const required = {
    enabled: true, contrast: 1.4, lightness: 0.05, exposure: 0.25, saturation: 0.8,
    gamma: 1.1, blur: 0, sharpen: 0.3, strength: 0.7, device: "auto",
};
const optional = {
    preset: "Manual", texture: 0.6, clarity: 0.2, detail_scale: 2, sharpen_radius: 1.5,
    detail_threshold: 0.005, halo_suppression: 0.8, luminance_only: true,
};
const defaults = { ...required, ...optional };
const names = Object.keys(defaults);
const definition = {
    name: "MAIImagePostEffects",
    input: {
        required: { image: ["IMAGE"], ...required },
        optional: { ...optional, effect_mask: ["MASK"] },
    },
};
const visibleInPreset = new Set(["preset", "enabled", "strength", "detail_scale", "device"]);

class BaseNode {
    constructor() {
        this.size = [420, 1000];
        this.inputs = [{ name: "image", link: 1 }, { name: "effect_mask", link: 2 }];
        this.outputs = [{ name: "image", type: "IMAGE" }];
        this.widgets = Object.entries(defaults).map(([name, value]) => ({
            name, value, type: "number", callbackCalls: 0,
            callback() { this.callbackCalls += 1; return "callback"; },
            serializeValue(node, index) {
                assert.equal(names[index], this.name, "Serializer must see the released order");
                return this.value;
            },
        }));
    }
    computeSize() { return [300, 80 + this.widgets.filter((widget) => !widget.hidden).length * 24]; }
    setSize(size) { this.size = size; }
    setDirtyCanvas() {}
    onNodeCreated() { return "created"; }
    onConfigure() { return "configured"; }
    configure(config) {
        config.widgets_values?.forEach((value, index) => {
            this.widgets[index].value = value;
            // Callback-driven layout changes must not disturb restoration order.
            this.widgets[index].callback?.(value);
        });
        if (config.size) this.size = [...config.size];
        return this.onConfigure(config);
    }
    serialize() {
        if (this.failSerialization) throw new Error("Serialization failed");
        return { widgets_values: this.widgets.map((widget, index) => widget.serializeValue(this, index)) };
    }
}
class Node extends BaseNode {}
const original = Node.prototype.onNodeCreated;
extension.beforeRegisterNodeDef(Node, { name: "OtherNode" });
assert.equal(Node.prototype.onNodeCreated, original);
extension.beforeRegisterNodeDef(Node, definition);

const node = new Node();
const widgetArray = node.widgets;
const slots = JSON.stringify([node.inputs, node.outputs]);
const widget = (name, target = node) => target.widgets.find((item) => item.name === name);
assert.equal(node.onNodeCreated(), "created");
await Promise.resolve();
assert.equal(node.widgets[0].name, "preset");
assert.equal(node.widgets, widgetArray);
assert.ok(node.widgets.every((item) => !item.hidden));
const fullHeight = node.size[1];

const preset = widget("preset");
preset.value = "Prepare for refinement";
assert.equal(preset.callback(preset.value), "callback");
assert.equal(preset.callbackCalls, 1);
for (const item of node.widgets) {
    assert.equal(Boolean(item.hidden), !visibleInPreset.has(item.name), item.name);
}
assert.ok(node.size[1] < fullHeight);
assert.equal(node.size[0], 420);
assert.equal(JSON.stringify([node.inputs, node.outputs]), slots);
const saved = node.serialize();
assert.deepEqual(saved.widgets_values, names.map((name) => widget(name).value));
assert.equal(saved.widgets_values[10], "Prepare for refinement");
assert.equal(node.widgets[0].name, "preset");

// Save/load and repeated loads retain every manual value, even when hidden.
const restored = new Node();
restored.onNodeCreated();
await Promise.resolve();
for (let count = 0; count < 2; count += 1) {
    assert.equal(restored.configure({ ...saved, size: [500, 900] }), "configured");
    await Promise.resolve();
    assert.equal(restored.widgets[0].name, "preset");
    assert.deepEqual(restored.serialize(), saved);
    assert.equal(widget("contrast", restored).value, 1.4);
    assert.equal(widget("texture", restored).value, 0.6);
    assert.equal(widget("contrast", restored).hidden, true);
    assert.equal(restored.size[0], 500);
    assert.ok(restored.size[1] < 900);
}
widget("preset", restored).value = "Manual";
widget("preset", restored).callback("Manual");
assert.ok(restored.widgets.every((item) => !item.hidden));
assert.equal(widget("texture", restored).value, 0.6);

// The oldest workflows have no appended detail/preset values at all.
const legacy = new Node();
legacy.onNodeCreated();
await Promise.resolve();
legacy.configure({ widgets_values: Object.values(required) });
await Promise.resolve();
assert.equal(widget("preset", legacy).value, "Manual");
assert.equal(widget("enabled", legacy).value, true);
assert.equal(widget("device", legacy).value, "auto");
assert.equal(legacy.widgets[0].name, "preset");
assert.ok(legacy.widgets.every((item) => !item.hidden));

// Canonical saved arrays also load correctly with the extension unavailable.
const bare = new BaseNode();
bare.configure(saved);
assert.equal(widget("preset", bare).value, "Prepare for refinement");
assert.equal(widget("contrast", bare).value, 1.4);

// Connected presets are unknown until execution, so leave all controls available.
node.inputs.push({ name: "preset", widget: { name: "preset" }, link: 42 });
node.onConnectionsChange();
await Promise.resolve();
assert.ok(node.widgets.every((item) => !item.hidden));
node.inputs.pop();
node.onConnectionsChange();
await Promise.resolve();
assert.equal(widget("contrast").hidden, true);

// Converted controls keep their widget state and input socket in both presets.
widget("contrast").type = "converted-widget";
node.inputs.push({ name: "contrast", widget: { name: "contrast" }, link: 43 });
node.onConnectionsChange();
await Promise.resolve();
assert.equal(widget("contrast").hidden, undefined);
assert.equal(widget("contrast").type, "converted-widget");
assert.equal(node.inputs.at(-1).link, 43);

// Rebinding lifecycle hooks does not wrap the preset callback repeatedly.
node.onConnectionsChange();
node.onConfigure({});
await Promise.resolve();
const callbackCount = preset.callbackCalls;
preset.value = "Manual";
preset.callback("Manual");
assert.equal(preset.callbackCalls, callbackCount + 1);
assert.equal(widget("lightness").value, 0.05);

// Even failed saves leave the live widget layout intact.
node.failSerialization = true;
assert.throws(() => node.serialize(), /Serialization failed/);
assert.equal(node.widgets[0].name, "preset");
assert.equal(node.widgets, widgetArray);
assert.equal(node.inputs.at(-1).link, 43);
console.log("Post effects preset ordering, visibility, legacy load, save/reload and sockets passed.");
