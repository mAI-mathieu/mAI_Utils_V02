import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const draws = [];
const context = new Proxy({}, { get(target, name) {
    return target[name] ?? ((...args) => { if (name === "drawImage") draws.push(args[0].src); });
} });
class Element {
    constructor(tag) { this.tag = tag; this.style = {}; this.children = []; }
    append(...children) { this.children.push(...children); }
    setAttribute(name, value) { this[name] = value; }
    addEventListener() {}
    getContext() { return context; }
    getBoundingClientRect() { return { left: 0, width: 800, height: 450 }; }
    remove() { this.removed = true; }
}
const listeners = new Map();
globalThis.document = {
    hidden: false,
    createElement: (tag) => new Element(tag),
    createTextNode: (text) => text,
    addEventListener: (name, handler) => listeners.set(name, handler),
    removeEventListener: (name, handler) => { if (listeners.get(name) === handler) listeners.delete(name); },
};
let clock = 0;
Object.defineProperty(globalThis, "performance", { value: { now: () => clock }, configurable: true });
let rafId = 0;
const rafs = new Map();
globalThis.requestAnimationFrame = (handler) => { rafs.set(++rafId, handler); return rafId; };
globalThis.cancelAnimationFrame = (id) => rafs.delete(id);
const images = [];
let automatic = true;
globalThis.Image = class {
    constructor() { this.width = 800; this.height = 450; }
    set src(value) {
        this.url = value;
        images.push(this);
        if (automatic && value) queueMicrotask(() => this.onload?.());
    }
    get src() { return this.url; }
};
const settle = async () => { for (let i = 0; i < 12; i++) await Promise.resolve(); };
async function advance(milliseconds) {
    clock = milliseconds;
    const work = [...rafs.values()];
    rafs.clear();
    for (const handler of work) await handler(clock);
    await settle();
}

let extension;
globalThis.testApp = { registerExtension(value) { extension = value; } };
globalThis.testApi = { apiURL: (url) => `/comfy${url}` };
const source = (await readFile(new URL("../web/js/video_compare.js", import.meta.url), "utf8"))
    .replace('import { app } from "../../../scripts/app.js";', "const app = globalThis.testApp;")
    .replace('import { api } from "../../../scripts/api.js";', "const api = globalThis.testApi;");
const { VideoComparePlayer } = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
const data = { subfolder: "mai_compare_test_ready", frame_count: 5, source_counts: [5, 8], fps: 10, width: 800, height: 450 };
const owner = { properties: {} };
const player = new VideoComparePlayer(owner);
assert.equal(player.playButton.disabled, true);
player.setData(data);
await settle();
assert.equal(player.index, 0);
assert.equal(player.playButton.disabled, false);
assert.match(player.message.textContent, /ends at shorter sequence/);
assert.match(player.pair[0].src, /^\/comfy\/view\?/);
assert.match(player.pair[0].src, /filename=a_000000.png/);
assert.match(player.pair[1].src, /filename=b_000000.png/);
player.canvas.onpointermove({ clientX: 200 });
assert.equal(player.reveal, 0.25);
player.setReveal(-10);
assert.equal(player.reveal, 0);
player.setReveal(10);
assert.equal(player.reveal, 1);
player.revealSlider.value = "40";
player.revealSlider.oninput();
assert.equal(player.reveal, 0.4);

// One clock selects both frames; looping and non-looping endpoints agree.
player.playButton.onclick();
assert.equal(player.playing, true);
await advance(200);
assert.equal(player.index, 2);
assert.match(player.pair[0].src, /a_000002/);
assert.match(player.pair[1].src, /b_000002/);
await advance(600);
assert.equal(player.index, 1);
player.loop.checked = false;
player.loop.onchange();
assert.equal(owner.properties.maiVideoCompareLoop, false);
await advance(900);
assert.equal(player.index, 4);
assert.equal(player.playing, false);
assert.equal(rafs.size, 0);
player.play();
await advance(900);
assert.equal(player.index, 0);
player.timeline.value = "3";
player.timeline.oninput();
await settle();
assert.equal(player.index, 3);
assert.equal(player.playing, false);

// B decoding late must never display A from a different frame.
automatic = false;
player.setData({ ...data, subfolder: "delayed" });
const pair = images.slice(-2);
pair[0].onload();
await settle();
assert.equal(player.pair, null);
pair[1].onload();
await settle();
assert.match(player.pair[0].src, /a_000000/);
assert.match(player.pair[1].src, /b_000000/);

// Superseded requests cannot overwrite a newer seek or dataset.
const pending = player.showFrame(4);
const stale = images.slice(-2);
player.setData({ ...data, subfolder: "replacement" });
const fresh = images.slice(-2);
stale.forEach((image) => image.onload?.());
await pending;
assert.equal(player.pair, null);
fresh.forEach((image) => image.onload());
await settle();
assert.match(player.pair[0].src, /subfolder=replacement/);

// Errors remain visible, stop playback, and do not create extra timers.
const failed = player.showFrame(4);
images.at(-1).onerror();
await failed;
assert.match(player.message.textContent, /queue again/);
assert.equal(player.playing, false);
automatic = true;
player.setData({ ...data, subfolder: "long", frame_count: 60, source_counts: [60, 60] });
await settle();
for (let i = 0; i < 60; i++) await player.showFrame(i);
assert.ok(player.cache.size <= 12);
player.play();
document.hidden = true;
listeners.get("visibilitychange")();
assert.equal(player.playing, false);
player.dispose();
assert.equal(player.cache.size, 0);
assert.equal(rafs.size, 0);
assert.equal(listeners.size, 0);
assert.equal(player.root.removed, true);

// Extension hooks preserve existing callbacks and create only our target widget.
class Node {
    constructor() { this.size = [100, 100]; this.properties = {}; }
    onNodeCreated() { return "created"; }
    onExecuted() { return "executed"; }
    onConfigure() { return "configured"; }
    onRemoved() { return "removed"; }
    addDOMWidget(...args) { this.widgetArgs = args; return {}; }
    setSize(size) { this.size = size; }
}
const original = Node.prototype.onNodeCreated;
extension.beforeRegisterNodeDef(Node, { name: "Other" });
assert.equal(Node.prototype.onNodeCreated, original);
extension.beforeRegisterNodeDef(Node, { name: "MAIVideoCompare" });
const node = new Node();
assert.equal(node.onNodeCreated(), "created");
assert.equal(node.widgetArgs[3].serialize, false);
node.properties.maiVideoCompareLoop = false;
assert.equal(node.onConfigure(), "configured");
assert.equal(node.maiVideoCompare.loop.checked, false);
assert.equal(node.onExecuted({ video_compare: [data] }), "executed");
await settle();
globalThis.testApp.rootGraph = { getNodeById: (id) => id === "7" ? node : undefined };
extension.onNodeOutputsUpdated({ "7": { video_compare: [{ ...data, subfolder: "history" }] }, "8": {} });
await settle();
assert.equal(node.maiVideoCompare.data.subfolder, "history");
const restoredPair = node.maiVideoCompare.pair;
extension.onNodeOutputsUpdated({ "7": { video_compare: [{ ...data, subfolder: "history" }] } });
assert.equal(node.maiVideoCompare.pair, restoredPair);
assert.equal(node.onRemoved(), "removed");
assert.equal(listeners.size, 0);
console.log("Synchronized pairs, delayed decoding, hover, seek, play/pause, loop, errors, bounded cache and cleanup passed.");
