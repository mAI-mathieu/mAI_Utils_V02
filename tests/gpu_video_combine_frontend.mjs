import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

class Element {
    constructor(tag) {
        this.tagName = tag.toUpperCase();
        this.children = [];
        this.dataset = {};
        this.muted = false;
    }
    append(...children) {
        for (const child of children) {
            child.parent?.removeChild(child);
            this.children.push(child);
            child.parent = this;
        }
    }
    removeChild(child) {
        this.children = this.children.filter((item) => item !== child);
        child.parent = null;
    }
    replaceChildren(...children) {
        for (const child of [...this.children]) this.removeChild(child);
        this.append(...children);
    }
    contains(child) { return child === this || this.children.some((item) => item.contains(child)); }
    matches(selector) { return selector === "video" ? this.tagName === "VIDEO" : this.dataset.nodeId != null; }
    closest(selector) { return this.matches(selector) ? this : this.parent?.closest(selector); }
    querySelectorAll(selector) {
        return this.children.flatMap((child) => [
            ...(child.matches(selector) ? [child] : []), ...child.querySelectorAll(selector),
        ]);
    }
}

const root = new Element("document");
const listeners = new Map();
root.addEventListener = (name, handler, capture) => {
    assert.equal(capture, true);
    assert.ok(!listeners.has(name), "Only one shared media listener is needed");
    listeners.set(name, handler);
};
root.removeEventListener = (name, handler, capture) => {
    assert.equal(capture, true);
    assert.equal(listeners.get(name), handler);
    listeners.delete(name);
};
const emit = (type, target) => listeners.get(type)?.({ target });
const observers = [];
globalThis.MutationObserver = class {
    constructor(callback) { this.callback = callback; observers.push(this); }
    observe(target, options) {
        assert.equal(target, root);
        assert.deepEqual(options, { childList: true, subtree: true });
    }
    disconnect() { this.disconnected = true; }
    added(...nodes) { this.callback([{ addedNodes: nodes }]); }
};
globalThis.document = root;
let changes = 0;
const graph = { change() { changes++; } };
let extension;
globalThis.testApp = { canvas: { graph }, registerExtension(value) { extension = value; } };
const source = (await readFile(new URL("../web/js/gpu_video_combine.js", import.meta.url), "utf8"))
    .replace('import { app } from "../../../scripts/app.js";', "const app = globalThis.testApp;");
const { VideoPreviewMuteMemory } = await import(`data:text/javascript;base64,${Buffer.from(source).toString("base64")}`);

const node = { id: 1, graph, properties: {}, videoContainer: new Element("div") };
const video = new Element("video");
node.videoContainer.append(video);
root.append(node.videoContainer);
const memory = new VideoPreviewMuteMemory(testApp, root);
memory.watch(node);
memory.watch(node);
assert.equal(observers.length, 1);
assert.equal(video.muted, false, "A fresh node keeps the native player default");
assert.deepEqual(node.properties, {});

// Muting survives asynchronous replacement of the native canvas preview.
video.muted = true;
emit("volumechange", video);
assert.equal(node.properties.maiVideoPreviewMuted, true);
assert.equal(changes, 1);
const replacement = new Element("video");
node.videoContainer.replaceChildren(replacement);
observers[0].added(replacement);
assert.equal(replacement.muted, true);
emit("volumechange", replacement); // Browser event queued by restoration.
assert.equal(changes, 1);
assert.equal(node.properties.maiVideoPreviewMuted, true);
const earlyEvent = new Element("video");
node.videoContainer.replaceChildren(earlyEvent);
emit("volumechange", earlyEvent);
assert.equal(earlyEvent.muted, true, "Initial media events must restore the saved setting");
assert.equal(node.properties.maiVideoPreviewMuted, true);
// A late event from a detached old player must not change the preference.
video.muted = false;
emit("volumechange", video);
assert.equal(node.properties.maiVideoPreviewMuted, true);

// Explicit unmute (false) is retained too, including reused source elements.
earlyEvent.muted = false;
emit("volumechange", earlyEvent);
const next = new Element("video");
next.muted = true;
node.videoContainer.replaceChildren(next);
observers[0].added(next);
assert.equal(next.muted, false);
next.muted = true;
emit("loadedmetadata", next);
assert.equal(next.muted, false);

// Widget roots cover native previews before videoContainer is exposed.
const widgetRoot = new Element("div");
const widgetVideo = new Element("video");
widgetRoot.append(widgetVideo);
root.append(widgetRoot);
node.widgets = [{ element: widgetRoot }];
node.properties.maiVideoPreviewMuted = true;
observers[0].added(widgetRoot);
assert.equal(widgetVideo.muted, true);

// Vue nodes use a node-ID ancestor and stay independent of other nodes/graphs.
const vueNode = { id: 2, graph, properties: { maiVideoPreviewMuted: false } };
memory.watch(vueNode);
const vueRoot = new Element("div");
vueRoot.dataset.nodeId = "2";
const vueVideo = new Element("video");
vueVideo.muted = true;
vueRoot.append(vueVideo);
root.append(vueRoot);
observers[0].added(vueRoot);
assert.equal(vueVideo.muted, false);
vueVideo.muted = true;
emit("volumechange", vueVideo);
assert.equal(vueNode.properties.maiVideoPreviewMuted, true);
assert.equal(node.properties.maiVideoPreviewMuted, true);
vueVideo.muted = false;
emit("volumechange", vueVideo);
assert.equal(vueNode.properties.maiVideoPreviewMuted, false);
assert.equal(node.properties.maiVideoPreviewMuted, true);

const unrelatedRoot = new Element("div");
unrelatedRoot.dataset.nodeId = "999";
const unrelated = new Element("video");
unrelatedRoot.append(unrelated);
root.append(unrelatedRoot);
observers[0].added(unrelatedRoot);
unrelated.muted = true;
emit("volumechange", unrelated);
emit("loadedmetadata", unrelated);
assert.equal(unrelated.muted, true);
const hiddenNode = { id: 999, graph: {}, properties: { maiVideoPreviewMuted: false } };
memory.watch(hiddenNode);
observers[0].added(unrelatedRoot);
assert.equal(unrelated.muted, true, "Same IDs in inactive graphs must not match");
memory.unwatch(hiddenNode);

// Saved node properties restore after workflow load; non-video mutations are safe.
assert.equal(JSON.parse(JSON.stringify({ properties: node.properties })).properties.maiVideoPreviewMuted, true);
observers[0].added({});
memory.unwatch(node);
assert.equal(listeners.size, 2);
memory.unwatch(vueNode);
assert.equal(listeners.size, 0);
assert.equal(observers[0].disconnected, true);

// Hook only the target class, preserve existing callbacks, and release observers.
class Node {
    constructor() {
        this.id = 3;
        this.graph = graph;
        this.properties = { maiVideoPreviewMuted: true };
        this.videoContainer = new Element("div");
        this.video = new Element("video");
        this.videoContainer.append(this.video);
        root.append(this.videoContainer);
    }
    onNodeCreated() { return "created"; }
    onConfigure() { this.properties.maiVideoPreviewMuted = false; return "configured"; }
    onRemoved() { return "removed"; }
}
const original = Node.prototype.onNodeCreated;
extension.beforeRegisterNodeDef(Node, { name: "OtherNode" });
assert.equal(Node.prototype.onNodeCreated, original);
extension.beforeRegisterNodeDef(Node, { name: "MAIGPUVideoCombine" });
const owner = new Node();
assert.equal(owner.onNodeCreated(), "created");
assert.equal(owner.video.muted, true);
assert.equal(owner.onConfigure(), "configured");
assert.equal(owner.video.muted, false);
assert.equal(owner.onRemoved(), "removed");
assert.equal(listeners.size, 0);
assert.equal(observers.at(-1).disconnected, true);
console.log("GPU video combine frontend: mute/unmute survives reruns, source changes and workflow load; node isolation and cleanup pass.");
