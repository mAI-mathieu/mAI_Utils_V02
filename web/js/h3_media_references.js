import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const BASE = "/mai/h3/references";
const DRAG_TYPE = "application/x-mai-h3-reference";
let settingsPromise;
let panelSequence = 0;
const SORT_OPTIONS = {
    name_asc: "Name (A–Z)",
    name_desc: "Name (Z–A)",
    type: "Type, then name (A–Z)",
    manual: "Manual order",
};
const nameCollator = new Intl.Collator(undefined, { numeric: true, sensitivity: "base" });

export function sortReferenceItems(items, mode = "name_asc") {
    if (!Object.hasOwn(SORT_OPTIONS, mode)) throw new Error(`Unknown reference sort order: ${mode}`);
    const sorted = [...items];
    if (mode === "manual") return sorted;
    const typeOrder = { image: 0, video: 1, audio: 2 };
    return sorted.sort((a, b) => {
        if (mode === "type") {
            const byType = typeOrder[a.kind] - typeOrder[b.kind];
            if (byType) return byType;
        }
        const byName = nameCollator.compare(a.name, b.name);
        return mode === "name_desc" ? -byName : byName;
    });
}

function element(tag, text, style = "") {
    const el = document.createElement(tag);
    if (text) el.textContent = text;
    el.style.cssText = style;
    return el;
}

async function settings() {
    if (!settingsPromise) {
        settingsPromise = api.fetchApi(`${BASE}/settings`).then(async (response) => {
            if (!response.ok) throw new Error("Reference routes unavailable. Restart ComfyUI and refresh this page.");
            return response.json();
        }).catch((error) => { settingsPromise = null; throw error; });
    }
    return settingsPromise;
}

// Capture entries during the drop event; browsers restrict DataTransfer later.
async function droppedFiles(transfer) {
    const entries = Array.from(transfer.items ?? [], (item) => item.webkitGetAsEntry?.());
    const fallback = Array.from(transfer.files ?? []);
    if (!entries.length || entries.some((entry) => !entry)) return fallback;
    async function visit(entry) {
        if (entry.isFile) return [await new Promise((resolve, reject) => entry.file(resolve, reject))];
        const reader = entry.createReader();
        const children = [];
        while (true) {
            const batch = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
            if (!batch.length) break;
            children.push(...batch);
        }
        children.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
        const files = [];
        for (const child of children) files.push(...await visit(child));
        return files;
    }
    const files = [];
    for (const entry of entries) files.push(...await visit(entry));
    return files;
}

export class H3ReferencePanel {
    constructor(node, widget) {
        this.node = node;
        this.widget = widget;
        this.items = [];
        // This only distinguishes live panels during drag-and-drop. A local
        // counter also works on HTTP LAN pages, where randomUUID is unavailable.
        this.token = `h3-panel-${++panelSequence}`;
        this.abort = new AbortController();
        this.root = element("div", "", "display:flex;flex-direction:column;gap:8px;width:100%;height:100%;min-height:330px;padding:10px;box-sizing:border-box;background:#181a20;color:#eef0f5;border:1px solid #454956;border-radius:8px;font:12px sans-serif;overflow:hidden;");
        const toolbar = element("div", "", "display:flex;gap:6px;flex-wrap:wrap;flex-shrink:0;");
        this.files = this.picker(false);
        this.folder = this.picker(true);
        this.buttons = [];
        toolbar.append(this.button("Add files", () => this.files.click()),
            this.button("Choose folder", () => this.folder.click()),
            this.button("Clear", () => { this.items = []; this.save(); this.render(); }));
        const sortRow = element("label", "Sort", "display:flex;align-items:center;gap:8px;flex-shrink:0;");
        this.sortSelect = element("select", "", "flex:1;min-width:0;background:#2a2e38;color:inherit;border:1px solid #5b6270;border-radius:4px;padding:4px;");
        this.sortSelect.setAttribute("aria-label", "Reference sort order");
        for (const [value, label] of Object.entries(SORT_OPTIONS)) {
            const option = element("option", label);
            option.value = value;
            this.sortSelect.append(option);
        }
        this.sortSelect.addEventListener("change", () => this.setSortMode(this.sortSelect.value));
        sortRow.append(this.sortSelect);
        this.count = element("div", "", "font-weight:600;flex-shrink:0;");
        this.list = element("div", "", "flex:1;min-height:140px;overflow:auto;display:flex;flex-direction:column;gap:8px;border:1px dashed #5b6270;border-radius:6px;padding:6px;");
        this.message = element("div", "Drop media here. Drag the grip to change output order.", "line-height:1.4;flex-shrink:0;overflow-wrap:anywhere;");
        this.message.setAttribute("role", "status");
        this.root.append(toolbar, sortRow, this.count, this.list, this.message, this.files, this.folder);
        this.root.append(element("div", "Wire matching ref_* sockets to MiniMax H3 Reference to Video. Order is numbered separately for images, videos and audio.", "font-size:11px;line-height:1.4;color:#b9c4d8;flex-shrink:0;"));
        for (const name of ["pointerdown", "mousedown", "keydown", "wheel"]) {
            this.root.addEventListener(name, (event) => event.stopPropagation());
        }
        this.root.addEventListener("dragover", (event) => {
            event.preventDefault();
            event.stopPropagation();
            event.dataTransfer.dropEffect = event.dataTransfer.types.includes(DRAG_TYPE) ? "move" : "copy";
        });
        this.root.addEventListener("drop", (event) => {
            event.preventDefault();
            event.stopPropagation();
            if (this.busy) return;
            const payload = event.dataTransfer.getData(DRAG_TYPE);
            if (payload) { this.reorder(payload, this.items.length); return; }
            // Start traversing the directory now, before the event's transfer expires.
            const files = droppedFiles(event.dataTransfer);
            this.ingest(files);
        });
        this.restore();
        settings().then((config) => {
            const accept = Object.values(config.extensions).flat().join(",");
            this.files.accept = accept;
            // Folder inputs intentionally allow unrelated files; ingest filters them.
        }).catch((error) => this.status(error.message, true));
    }

    button(label, action, title = label) {
        const button = element("button", label, "cursor:pointer;padding:4px 9px;border:1px solid #5b6270;border-radius:4px;background:#2a2e38;color:inherit;");
        button.type = "button";
        button.title = title;
        button.setAttribute("aria-label", title);
        button.addEventListener("click", action);
        this.buttons.push(button);
        return button;
    }

    picker(folder) {
        const input = document.createElement("input");
        input.type = "file";
        input.multiple = true;
        input.hidden = true;
        if (folder) { input.webkitdirectory = true; input.setAttribute("webkitdirectory", ""); }
        input.addEventListener("change", () => {
            const files = Array.from(input.files ?? []);
            input.value = "";
            this.ingest(Promise.resolve(files));
        });
        return input;
    }

    status(message, error = false) {
        if (this.disposed) return;
        this.message.textContent = message;
        this.message.style.color = error ? "#ffb4ab" : "#b9c4d8";
    }

    save() {
        this.widget.value = JSON.stringify(this.items);
        this.node.properties ??= {};
        this.node.properties.maiH3ReferenceSort = this.sortMode;
        this.node.graph?.change?.();
        this.node.setDirtyCanvas(true, true);
    }

    restore() {
        const mode = this.node.properties?.maiH3ReferenceSort;
        this.sortMode = Object.hasOwn(SORT_OPTIONS, mode) ? mode : "name_asc";
        this.sortSelect.value = this.sortMode;
        try {
            const items = JSON.parse(this.widget.value || "[]");
            if (!Array.isArray(items) || items.some((item) => !item || !/^[a-f0-9]{64}\.[a-z0-9]+$/.test(item.filename)
                || !["image", "audio", "video"].includes(item.kind) || typeof item.name !== "string")) {
                throw new Error("Invalid saved media manifest. Repair the JSON or clear the references.");
            }
            this.items = items;
            this.render();
        } catch (error) {
            // Preserve invalid workflow data so it can be repaired rather than lost.
            this.items = [];
            this.render();
            this.status(error.message, true);
        }
    }

    releaseMedia() {
        for (const media of this.list.querySelectorAll("video, audio")) {
            media.pause();
            media.removeAttribute("src");
            media.load();
        }
    }

    render() {
        this.releaseMedia();
        this.list.replaceChildren();
        // Rebuild the card buttons without retaining detached DOM elements.
        this.buttons = this.buttons.slice(0, 3);
        const counts = { image: 0, audio: 0, video: 0 };
        this.items.forEach((item) => { counts[item.kind]++; });
        this.count.textContent = `${this.items.length} references · ${counts.image} images · ${counts.audio} audio · ${counts.video} videos`;
        if (!this.items.length) this.list.append(element("div", "Drop pictures, audio or videos here, or choose a folder. Folder imports include subfolders; unsupported files are skipped.", "margin:auto;padding:18px;color:#b9c4d8;text-align:center;line-height:1.6;"));
        this.items.forEach((item, index) => {
            const slot = this.items.slice(0, index).filter((other) => other.kind === item.kind).length;
            const card = element("div", "", "background:#252934;border:1px solid #4a5060;border-radius:6px;padding:7px;flex-shrink:0;");
            const heading = element("div", "", "display:flex;align-items:center;gap:6px;margin-bottom:6px;");
            const grip = element("span", "⠿", "cursor:grab;font-size:22px;padding:0 6px;user-select:none;");
            grip.draggable = true;
            grip.title = "Drag to reorder";
            grip.addEventListener("dragstart", (event) => {
                if (this.busy) { event.preventDefault(); return; }
                event.stopPropagation();
                event.dataTransfer.setData(DRAG_TYPE, `${this.token}:${index}`);
                event.dataTransfer.effectAllowed = "move";
                event.dataTransfer.setDragImage(card, 15, 15);
            });
            const label = element("span", `ref_${item.kind}_${slot} · ${item.name}`, "flex:1;min-width:0;overflow-wrap:anywhere;");
            label.title = item.kind === "video" ? `Also outputs ref_video_audio_${slot} when a soundtrack exists.` : `Connect to ref_${item.kind}_${slot}`;
            const up = this.button("↑", () => this.move(index, index - 1), `Move ${item.name} up`);
            const down = this.button("↓", () => this.move(index, index + 1), `Move ${item.name} down`);
            up.disabled = index === 0 || this.busy;
            down.disabled = index === this.items.length - 1 || this.busy;
            const remove = this.button("×", () => {
                this.items.splice(index, 1); this.save(); this.render();
            }, `Remove ${item.name}`);
            remove.disabled = Boolean(this.busy);
            heading.append(grip, label, up, down, remove);
            const preview = element(item.kind === "image" ? "img" : item.kind, "", "display:block;width:100%;max-height:160px;object-fit:contain;border-radius:4px;background:#101218;");
            if (item.kind === "image") { preview.alt = item.name; preview.loading = "lazy"; }
            else { preview.controls = true; preview.preload = "metadata"; preview.playsInline = true; }
            preview.draggable = false;
            const error = element("div", "", "color:#ffb4ab;font-size:11px;line-height:1.4;");
            preview.addEventListener("error", () => {
                if (preview.getAttribute("src")) error.textContent = "Preview unavailable: the file may be missing, damaged, or use a codec this browser cannot play.";
            });
            preview.src = api.apiURL(`${BASE}/media/${encodeURIComponent(item.filename)}`);
            card.append(heading, preview, error);
            card.addEventListener("dragover", (event) => {
                if (!event.dataTransfer.types.includes(DRAG_TYPE)) return;
                card.style.borderColor = "#a5c9ff";
            });
            card.addEventListener("dragleave", () => { card.style.borderColor = "#4a5060"; });
            card.addEventListener("drop", (event) => {
                const payload = event.dataTransfer.getData(DRAG_TYPE);
                if (!payload) return; // External file drops bubble to the panel.
                event.preventDefault(); event.stopPropagation();
                card.style.borderColor = "#4a5060";
                const box = card.getBoundingClientRect();
                this.reorder(payload, index + (event.clientY > box.top + box.height / 2 ? 1 : 0));
            });
            this.list.append(card);
        });
    }

    reorder(payload, insertion) {
        if (this.busy) return;
        const [token, value] = payload.split(":");
        const from = Number(value);
        if (token !== this.token || !Number.isInteger(from) || from < 0 || from >= this.items.length) return;
        this.move(from, insertion > from ? insertion - 1 : insertion);
    }

    move(from, to) {
        if (this.busy || to < 0 || to >= this.items.length || from === to) return;
        this.items.splice(to, 0, this.items.splice(from, 1)[0]);
        this.sortMode = "manual";
        this.sortSelect.value = this.sortMode;
        this.save(); this.render();
        this.status("Output order updated and saved in the workflow.");
    }

    setSortMode(mode) {
        if (this.busy) { this.sortSelect.value = this.sortMode; return; }
        this.items = sortReferenceItems(this.items, mode);
        this.sortMode = mode;
        this.sortSelect.value = mode;
        this.save(); this.render();
        this.status(mode === "manual" ? "Manual order selected. Drag the grip to reorder." : `Sorted by ${SORT_OPTIONS[mode]}. Output order saved.`);
    }

    async ingest(filesPromise) {
        if (this.busy || this.disposed) return;
        this.busy = true;
        this.sortSelect.disabled = true;
        for (const button of this.buttons) button.disabled = true;
        let added = 0;
        try {
            const files = await filesPromise;
            const config = await settings();
            const candidates = sortReferenceItems(files.map((file) => {
                const suffix = `.${file.name.split(".").pop().toLowerCase()}`;
                const kind = Object.keys(config.extensions).find((key) => config.extensions[key].includes(suffix));
                return { file, kind, name: file.name };
            }).filter((item) => item.kind), this.sortMode);
            const skipped = files.length - candidates.length;
            if (!candidates.length) throw new Error("No supported media found. Add PNG/JPEG/WebP/BMP, audio, or video files.");
            const counts = { image: 0, audio: 0, video: 0, total: 0 };
            for (const item of [...this.items, ...candidates]) { counts[item.kind]++; counts.total++; }
            for (const [kind, count] of Object.entries(counts)) {
                if (count > config.limits[kind]) throw new Error(`H3 allows ${config.limits[kind]} ${kind} references. Choose fewer files or remove existing cards.`);
            }
            for (const { file } of candidates) {
                if (!file.size || file.size > config.max_upload_bytes) throw new Error(`${file.name}: files must contain data and be at most ${config.max_upload_bytes / 1024 / 1024} MiB.`);
            }
            for (const { file } of candidates) {
                this.status(`Uploading ${added + 1}/${candidates.length}: ${file.name}`);
                const body = new FormData();
                body.append("file", file);
                const response = await api.fetchApi(`${BASE}/upload`, { method: "POST", body, signal: this.abort.signal });
                if (!response.ok) {
                    const detail = await response.text();
                    let message = detail;
                    try { message = JSON.parse(detail).error || detail; } catch { /* HTTP errors may be plain text. */ }
                    throw new Error(`${file.name}: ${message || `upload failed (${response.status})`}`);
                }
                if (this.disposed) return;
                this.items = sortReferenceItems([...this.items, await response.json()], this.sortMode);
                added++;
                this.save();
            }
            this.status(`Added ${added} reference(s).${skipped ? ` Skipped ${skipped} unsupported file(s).` : ""} Drag the grip to reorder.`);
        } catch (error) {
            this.status(`${added ? `Kept ${added} uploaded reference(s). ` : ""}${error.message}`, true);
        } finally {
            this.busy = false;
            if (!this.disposed) {
                this.sortSelect.disabled = false;
                for (const button of this.buttons.slice(0, 3)) button.disabled = false;
                this.render();
            }
        }
    }

    dispose() {
        this.disposed = true;
        this.abort.abort();
        this.releaseMedia();
        this.root.remove();
    }
}

app.registerExtension({
    name: "mAI.H3MediaReferences",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MAIH3MediaReferences") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            const manifest = this.widgets.find((widget) => widget.name === "media_manifest");
            if (!manifest) return result;
            // Keep the backend STRING widget serialized, while the panel edits it.
            manifest.type = "converted-widget";
            manifest.computeSize = () => [0, -4];
            if (manifest.inputEl) manifest.inputEl.hidden = true;
            manifest.options = { ...manifest.options, hidden: true };
            this.maiH3References = new H3ReferencePanel(this, manifest);
            const panel = this.addDOMWidget("h3_reference_panel", "mai_h3_references", this.maiH3References.root, {
                serialize: false, hideOnZoom: false, getMinHeight: () => 330, getHeight: () => 490,
            });
            panel.serialize = false;
            panel.serializeValue = () => undefined;
            const callback = manifest.callback;
            manifest.callback = (...values) => {
                callback?.apply(manifest, values);
                this.maiH3References?.restore();
            };
            this.setSize([Math.max(this.size[0], 470), Math.max(this.size[1], this.computeSize()[1], 700)]);
            return result;
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = configured?.apply(this, args);
            this.maiH3References?.restore();
            return result;
        };
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function (...args) {
            this.maiH3References?.dispose();
            return removed?.apply(this, args);
        };
    },
});
