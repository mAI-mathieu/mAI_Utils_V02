import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
// Plain HTTP LAN pages do not expose secure-context APIs such as randomUUID.
// Node creation must work there just as it does on localhost and HTTPS.
Object.defineProperty(globalThis, "crypto", { value: undefined, configurable: true });
class Element {
    constructor(tag) { this.tag = tag; this.style = {}; this.children = []; this.events = new Map(); }
    append(...children) { this.children.push(...children); }
    replaceChildren(...children) { this.children = children; }
    setAttribute(name, value) { this[name] = name === "webkitdirectory" ? true : value; }
    getAttribute(name) { return this[name]; }
    removeAttribute(name) { delete this[name]; }
    addEventListener(name, handler) { this.events.set(name, handler); }
    querySelectorAll(selector) {
        const tags = selector.split(",").map((tag) => tag.trim());
        return this.children.flatMap((child) => [...(tags.includes(child.tag) ? [child] : []), ...(child.querySelectorAll?.(selector) ?? [])]);
    }
    pause() { this.paused = true; }
    load() { this.loaded = true; }
    remove() { this.removed = true; }
    click() { this.clicked = true; }
    getBoundingClientRect() { return { top: 0, height: 100 }; }
}
globalThis.document = { createElement: (tag) => new Element(tag) };
const config = { extensions: { image: [".png", ".jpg"], video: [".mp4"], audio: [".wav"] },
    limits: { image: 9, video: 3, audio: 3, total: 15 }, max_upload_bytes: 512 * 1024 * 1024 };
const item = (number, kind = "image") => ({ filename: number.toString(16).padStart(64, "0") + { image: ".png", video: ".mp4", audio: ".wav" }[kind], name: `file ${number}`, kind });
let extension;
let uploads = [];
let failAt = Infinity;
globalThis.testApp = { registerExtension(value) { extension = value; } };
globalThis.testApi = {
    apiURL: (url) => `/comfy${url}`,
    async fetchApi(url, options) {
        if (url.endsWith("/settings")) return { ok: true, json: async () => config };
        assert.equal(options.method, "POST");
        uploads.push(options.body.get("file"));
        if (uploads.length === failAt) return { ok: false, status: 400, text: async () => JSON.stringify({ error: "test upload failure" }) };
        return { ok: true, json: async () => item(uploads.length) };
    },
};
const source = (await readFile(new URL("../web/js/h3_media_references.js", import.meta.url), "utf8"))
    .replace('import { app } from "../../../scripts/app.js";', "const app = globalThis.testApp;")
    .replace('import { api } from "../../../scripts/api.js";', "const api = globalThis.testApi;");
const { H3ReferencePanel } = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);
const settle = async () => { for (let i = 0; i < 25; i++) await Promise.resolve(); };
let dirty = 0;
const node = { graph: { change() { dirty++; } }, setDirtyCanvas() {} };
const widget = { value: JSON.stringify([item(1), item(2, "audio"), item(3), item(4, "video")]) };
const panel = new H3ReferencePanel(node, widget);
await settle();
assert.equal(panel.items.length, 4);
assert.equal(panel.folder.webkitdirectory, true);
assert.match(panel.files.accept, /\.mp4/);
assert.equal(panel.list.children[2].children[0].children[1].textContent, "ref_image_1 · file 3");
assert.match(panel.list.children[3].children[1].src, /^\/comfy\/mai\/h3\/references\/media\//);
// Reordering assigns the first image slot without changing the audio/video groups.
panel.reorder(`${panel.token}:2`, 0);
assert.deepEqual(panel.items.map((entry) => entry.name), ["file 3", "file 1", "file 2", "file 4"]);
assert.equal(JSON.parse(widget.value)[0].name, "file 3");
assert.match(panel.list.children[0].children[0].children[1].textContent, /^ref_image_0/);
panel.reorder("another-node:0", 4);
assert.equal(panel.items[0].name, "file 3");
// Drop below a card and drop at the panel end both use insertion positions.
panel.reorder(`${panel.token}:0`, 3);
assert.deepEqual(panel.items.map((entry) => entry.name), ["file 1", "file 2", "file 3", "file 4"]);
panel.reorder(`${panel.token}:0`, 4);
assert.equal(panel.items.at(-1).name, "file 1");
const reloaded = new H3ReferencePanel(node, widget);
assert.deepEqual(reloaded.items, panel.items);
assert.notEqual(reloaded.token, panel.token, "Separate panels must reject each other's drags without requiring browser crypto");
assert.ok(dirty >= 3);
// Remove only detaches the selected reference, preserving all others.
panel.list.children[0].children[0].children.at(-1).events.get("click")();
assert.equal(panel.items.length, 3);
assert.equal(panel.items[0].name, "file 3");
// Real FormData verifies multiple uploads, filtering, and useful partial errors.
const files = [new File(["a"], "A.png"), new File(["b"], "B.jpg"), new File(["c"], "notes.txt")];
await panel.ingest(Promise.resolve(files));
assert.equal(uploads.length, 2);
assert.match(panel.message.textContent, /Skipped 1 unsupported/);
failAt = 4;
await panel.ingest(Promise.resolve(files.slice(0, 2)));
assert.match(panel.message.textContent, /Kept 1 uploaded.*test upload failure/);
assert.equal(panel.busy, false);
// Limit failures occur before the first upload in a batch.
const before = uploads.length;
await panel.ingest(Promise.resolve(Array.from({ length: 10 }, () => new File(["x"], "x.png"))));
assert.equal(uploads.length, before);
assert.match(panel.message.textContent, /H3 allows 9 image/);
// Directory drop traversal includes every readEntries batch, including subfolders.
const fileEntry = (name) => ({ name, isFile: true, file(resolve) { resolve(new File(["x"], name)); } });
const directory = { name: "folder", isFile: false, createReader() {
    const batches = [[fileEntry("z.png")], [fileEntry("a.png")], []];
    return { readEntries(resolve) { resolve(batches.shift()); } };
} };
panel.items = [];
failAt = Infinity;
panel.root.events.get("drop")({ preventDefault() {}, stopPropagation() {}, dataTransfer: {
    getData() { return ""; }, items: [{ webkitGetAsEntry() { return directory; } }], files: [],
} });
await settle();
assert.equal(uploads.length, before + 2);
assert.equal(uploads.at(-2).name, "a.png");
assert.equal(uploads.at(-1).name, "z.png");
// Invalid saved data survives restore and presents a visible error.
widget.value = "broken";
panel.restore();
assert.equal(widget.value, "broken");
assert.match(panel.message.textContent, /JSON|Unexpected/);
panel.dispose();
reloaded.dispose();
assert.equal(panel.abort.signal.aborted, true);
assert.equal(panel.root.removed, true);
// Registration keeps the serialized manifest and restores it on workflow load.
class Node {
    constructor() {
        this.widgets = [{ name: "media_manifest", value: "[]", inputEl: new Element("textarea") }];
        this.size = [200, 100];
    }
    addDOMWidget(name, type, root, options) { return { name, root, options }; }
    computeSize() { return [470, 800]; }
    setSize(value) { this.size = value; }
    setDirtyCanvas() {}
}
extension.beforeRegisterNodeDef(Node, { name: "MAIH3MediaReferences" });
const registered = new Node();
registered.onNodeCreated();
assert.equal(registered.widgets[0].inputEl.hidden, true);
assert.equal(registered.widgets[0].value, "[]");
assert.equal(registered.size[1], 800);
registered.widgets[0].value = JSON.stringify([item(8)]);
registered.onConfigure();
assert.equal(registered.maiH3References.items[0].name, "file 8");
registered.onRemoved();
console.log("H3 media frontend checks passed");
