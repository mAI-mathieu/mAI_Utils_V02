import { app } from "../../../scripts/app.js";

const MUTE_PROPERTY = "maiVideoPreviewMuted";

export class VideoPreviewMuteMemory {
    constructor(application, root = document) {
        this.app = application;
        this.root = root;
        this.nodes = new Set();
        this.owners = new WeakMap();
        this.onLoaded = (event) => this.bindVideo(event.target, true);
        this.onVolumeChange = (event) => {
            const video = event.target;
            const node = this.findNode(video);
            if (!node) return;
            // A newly mounted player's initial volume event is not a user change.
            if (this.owners.get(video) !== node) {
                this.bindVideo(video);
                return;
            }
            node.properties ??= {};
            if (node.properties[MUTE_PROPERTY] !== video.muted) {
                node.properties[MUTE_PROPERTY] = video.muted;
                node.graph?.change?.();
            }
        };
    }

    findNode(video) {
        if (!video?.matches?.("video")) return;
        const nodeId = video.closest?.("[data-node-id]")?.dataset.nodeId;
        const graph = this.app.canvas?.graph ?? this.app.graph;
        for (const node of this.nodes) {
            // Canvas previews live in DOM widgets; Vue previews live under node IDs.
            if (node.videoContainer?.contains(video)
                || node.widgets?.some((widget) => widget.element?.contains?.(video))) return node;
            if (nodeId != null && String(node.id) === nodeId && (!graph || node.graph === graph)) return node;
        }
    }

    bindVideo(video, restore = false) {
        const node = this.findNode(video);
        if (!node) return;
        const newlyMounted = this.owners.get(video) !== node;
        this.owners.set(video, node);
        const muted = node.properties?.[MUTE_PROPERTY];
        if ((newlyMounted || restore) && typeof muted === "boolean" && video.muted !== muted) {
            video.muted = muted;
        }
    }

    scan(root, restore = false) {
        if (root.matches?.("video")) this.bindVideo(root, restore);
        for (const video of root.querySelectorAll?.("video") ?? []) this.bindVideo(video, restore);
    }

    watch(node) {
        if (this.nodes.has(node)) return;
        this.nodes.add(node);
        if (this.nodes.size === 1) {
            // Media events do not bubble. Capture also covers replacement video
            // elements and source changes without modifying ComfyUI's player.
            this.root.addEventListener("volumechange", this.onVolumeChange, true);
            this.root.addEventListener("loadedmetadata", this.onLoaded, true);
            this.observer = new MutationObserver((records) => {
                for (const record of records) {
                    for (const added of record.addedNodes) this.scan(added);
                }
            });
            this.observer.observe(this.root.documentElement ?? this.root, { childList: true, subtree: true });
        }
        this.scan(this.root);
    }

    unwatch(node) {
        this.nodes.delete(node);
        if (this.nodes.size) return;
        this.observer?.disconnect();
        this.root.removeEventListener("volumechange", this.onVolumeChange, true);
        this.root.removeEventListener("loadedmetadata", this.onLoaded, true);
        this.owners = new WeakMap();
    }
}

const muteMemory = new VideoPreviewMuteMemory(app);

app.registerExtension({
    name: "mAI.GPUVideoCombine.PreviewMute",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MAIGPUVideoCombine") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            muteMemory.watch(this);
            return result;
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = configured?.apply(this, args);
            muteMemory.scan(document, true);
            return result;
        };
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function (...args) {
            muteMemory.unwatch(this);
            return removed?.apply(this, args);
        };
    },
});
