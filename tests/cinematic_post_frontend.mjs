// Exercise the actual extension lifecycle without a browser or ComfyUI server.
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

let extension;
globalThis.testApp = { registerExtension(value) { extension = value; } };
const source = await readFile(new URL("../web/js/cinematic_post.js", import.meta.url), "utf8");
const importPath = source.match(/import \{ app \} from "([^"]+)";/)[1];
assert.equal(new URL(
    importPath, "http://localhost/extensions/mAI_Utils_V02/js/cinematic_post.js",
).pathname, "/scripts/app.js");
await import(`data:text/javascript;base64,${Buffer.from(source.replace(
    `import { app } from "${importPath}";`,
    "const app = globalThis.testApp;",
)).toString("base64")}`);
delete globalThis.testApp;

class Node {
    constructor() {
        this.size = [420, 1000];
        this.inputs = [{ name: "image", link: 10 }, { name: "subject_mask", link: 11 }];
        this.outputs = [{ name: "image", links: [12] }];
        this.widgets = [
            { name: "enabled", type: "toggle", value: true },
            { name: "preset", type: "combo", value: "Commercial Cinematic" },
            { name: "advanced_mode", type: "toggle", value: false,
                callback() { return "original callback"; } },
            ...["strength", "contrast", "highlight_rolloff", "saturation", "color_density",
                "halation_strength", "bloom_strength", "grain_strength", "vignette_strength",
                "exposure", "grain_seed", "grain_size"].map((name, value) => ({
                name, value, type: "number", serializeValue() { return this.value; },
            })),
            { name: "control_after_generate", type: "combo", value: "fixed" },
        ];
    }
    // Frontend 1.51.10 excludes hidden widgets from minimum-size calculation.
    computeSize() { return [300, 80 + this.widgets.filter((w) => !w.hidden).length * 24]; }
    setSize(size) { this.size = size; }
    setDirtyCanvas() { this.dirty = true; }
    onNodeCreated() { return "created"; }
    onConfigure(values) {
        this.widgets.forEach((widget, index) => { widget.value = values[index]; });
        this.size = [420, 1000];
        return "configured";
    }
}

const originalHook = Node.prototype.onNodeCreated;
extension.beforeRegisterNodeDef(Node, { name: "OtherNode" });
assert.equal(Node.prototype.onNodeCreated, originalHook);
extension.beforeRegisterNodeDef(Node, { name: "mAI_CinematicPost" });
const node = new Node();
const widgets = [...node.widgets];
const values = node.widgets.map((w) => w.value);
const slots = JSON.stringify([node.inputs, node.outputs]);
const exposure = node.widgets.find((w) => w.name === "exposure");
const serializeExposure = exposure.serializeValue;
const toggle = node.widgets.find((w) => w.name === "advanced_mode");
assert.equal(node.onNodeCreated(), "created");
await Promise.resolve();
const compactHeight = node.size[1];
assert.equal(node.size[0], 420);
assert.equal(node.widgets.filter((w) => !w.hidden).length, 12);
assert.equal(exposure.hidden, true);
assert.equal(node.widgets.at(-1).hidden, true); // Seed companion also hides.
assert.equal(exposure.type, "number");
assert.equal(exposure.serializeValue, serializeExposure);
assert.deepEqual(node.widgets.map((w) => w.value), values);
assert.equal(JSON.stringify([node.inputs, node.outputs]), slots);

toggle.value = true;
assert.equal(toggle.callback(true), "original callback");
assert.equal(node.widgets.filter((w) => !w.hidden).length, widgets.length);
assert.ok(node.size[1] > compactHeight);
assert.deepEqual(node.widgets, widgets); // Identity and ordering remain stable.

exposure.value = 0.75;
toggle.value = false;
toggle.callback(false);
const saved = node.widgets.map((w) => w.value);
assert.equal(node.onConfigure(saved), "configured");
await Promise.resolve();
assert.equal(node.size[1], compactHeight);
assert.equal(exposure.value, 0.75);
assert.equal(exposure.serializeValue(), 0.75);

// Reload an expanded workflow, even though creation initially used basic mode.
saved[2] = true;
node.onConfigure(saved);
await Promise.resolve();
assert.equal(exposure.hidden, undefined);
assert.ok(node.size[1] > compactHeight);

// Connections must stay accessible, including ones made while a widget is hidden.
toggle.value = false;
toggle.callback(false);
node.inputs.push({ name: "exposure", widget: { name: "exposure" }, link: 42 });
node.onConnectionsChange();
await Promise.resolve();
assert.equal(exposure.hidden, undefined);
assert.equal(node.inputs.at(-1).link, 42);
node.inputs.at(-1).link = null;
node.onConnectionsChange();
await Promise.resolve();
assert.equal(exposure.hidden, true);
toggle.value = true;
toggle.callback(true);
exposure.type = "converted-widget";
exposure.hidden = true;
toggle.value = false;
toggle.callback(false);
toggle.value = true;
toggle.callback(true);
assert.equal(exposure.type, "converted-widget");
assert.equal(exposure.hidden, true); // Another extension's visibility is retained.
console.log("Frontend visibility, resizing, persistence, hooks and sockets passed.");
