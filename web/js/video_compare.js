import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

function element(tag, style, text) {
    const result = document.createElement(tag);
    result.style.cssText = style;
    if (text) result.textContent = text;
    return result;
}

// Keep the current two frames in GPU textures. Moving the divider updates one
// shader uniform rather than uploading images or compositing pixels on CPU.
class GpuCompareRenderer {
    constructor(canvas) {
        this.gl = canvas.getContext("webgl", {
            alpha: false, antialias: false, powerPreference: "high-performance",
            failIfMajorPerformanceCaveat: true,
        });
        if (!this.gl) return;
        const gl = this.gl;
        const shader = (type, source) => {
            const result = gl.createShader(type);
            gl.shaderSource(result, source);
            gl.compileShader(result);
            if (!gl.getShaderParameter(result, gl.COMPILE_STATUS)) {
                const message = gl.getShaderInfoLog(result);
                gl.deleteShader(result);
                throw new Error(`Video compare GPU shader failed: ${message}`);
            }
            return result;
        };
        const vertex = shader(gl.VERTEX_SHADER, `
            attribute vec2 position;
            varying vec2 uv;
            void main() { uv = position * 0.5 + 0.5; gl_Position = vec4(position, 0.0, 1.0); }
        `);
        const fragment = shader(gl.FRAGMENT_SHADER, `
            precision highp float;
            varying vec2 uv;
            uniform sampler2D imageA;
            uniform sampler2D imageB;
            uniform vec2 fitA;
            uniform vec2 fitB;
            uniform vec2 canvasSize;
            uniform float reveal;
            vec3 sampleFrame(sampler2D image, vec2 fit) {
                vec3 background = vec3(8.0, 9.0, 11.0) / 255.0;
                vec2 local = (uv - 0.5) / fit + 0.5;
                if (any(lessThan(local, vec2(0.0))) || any(greaterThan(local, vec2(1.0)))) return background;
                vec4 pixel = texture2D(image, local);
                return mix(background, pixel.rgb, pixel.a);
            }
            void main() {
                vec3 color = uv.x <= reveal ? sampleFrame(imageA, fitA) : sampleFrame(imageB, fitB);
                vec2 distance = (uv - vec2(reveal, 0.5)) * canvasSize;
                if (abs(distance.x) <= max(1.0, canvasSize.x / 800.0)
                    || length(distance) <= max(5.0, canvasSize.x / 100.0)) color = vec3(1.0);
                gl_FragColor = vec4(color, 1.0);
            }
        `);
        this.program = gl.createProgram();
        gl.attachShader(this.program, vertex);
        gl.attachShader(this.program, fragment);
        gl.linkProgram(this.program);
        gl.deleteShader(vertex);
        gl.deleteShader(fragment);
        if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
            const message = gl.getProgramInfoLog(this.program);
            this.dispose();
            throw new Error(`Video compare GPU program failed: ${message}`);
        }
        this.buffer = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
        this.position = gl.getAttribLocation(this.program, "position");
        this.uniforms = Object.fromEntries(["imageA", "imageB", "fitA", "fitB", "canvasSize", "reveal"]
            .map((name) => [name, gl.getUniformLocation(this.program, name)]));
        this.textures = [gl.createTexture(), gl.createTexture()];
        for (const texture of this.textures) {
            gl.bindTexture(gl.TEXTURE_2D, texture);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        }
    }

    draw(canvas, pair, reveal) {
        const gl = this.gl;
        if (gl.isContextLost()) return;
        gl.viewport(0, 0, canvas.width, canvas.height);
        gl.clearColor(8 / 255, 9 / 255, 11 / 255, 1);
        gl.clear(gl.COLOR_BUFFER_BIT);
        if (!pair) { this.pair = null; return; }
        gl.useProgram(this.program);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
        gl.enableVertexAttribArray(this.position);
        gl.vertexAttribPointer(this.position, 2, gl.FLOAT, false, 0, 0);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
        gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
        pair.forEach((image, index) => {
            gl.activeTexture(gl.TEXTURE0 + index);
            gl.bindTexture(gl.TEXTURE_2D, this.textures[index]);
            if (this.pair !== pair) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
            const side = index === 0 ? "A" : "B";
            gl.uniform1i(this.uniforms[`image${side}`], index);
            const scale = Math.min(canvas.width / image.width, canvas.height / image.height);
            gl.uniform2f(this.uniforms[`fit${side}`], image.width * scale / canvas.width, image.height * scale / canvas.height);
        });
        this.pair = pair;
        gl.uniform2f(this.uniforms.canvasSize, canvas.width, canvas.height);
        gl.uniform1f(this.uniforms.reveal, reveal);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    dispose() {
        if (!this.gl) return;
        for (const texture of this.textures ?? []) this.gl.deleteTexture(texture);
        if (this.buffer) this.gl.deleteBuffer(this.buffer);
        if (this.program) this.gl.deleteProgram(this.program);
        this.pair = null;
    }
}

// One canvas commits both decoded images together. A single clock selects their
// shared frame index, so buffering/slow browsers can skip pairs but never drift.
export class VideoComparePlayer {
    constructor(node) {
        this.node = node;
        this.index = 0;
        this.reveal = 0.5;
        this.cache = new Map();
        this.request = 0;
        this.playing = false;
        this.playbackId = 0;
        this.disposed = false;
        this.root = element("div", "display:flex;flex-direction:column;gap:6px;width:100%;height:100%;min-height:260px;padding:8px;box-sizing:border-box;background:#181a20;color:#f3f4f6;font:12px sans-serif;border-radius:8px;overflow:hidden;");
        this.stage = element("div", "position:relative;flex:1;min-height:150px;background:#08090b;overflow:hidden;display:flex;align-items:center;justify-content:center;");
        this.canvas = element("canvas", "display:block;width:100%;height:100%;object-fit:contain;cursor:col-resize;touch-action:none;");
        this.canvas.width = 768;
        this.canvas.height = 432;
        this.canvas.tabIndex = 0;
        this.canvas.setAttribute("aria-label", "Video comparison. Move pointer to reveal A on the left and B on the right.");
        this.stage.append(this.canvas);
        for (const [name, side] of [["A", "left"], ["B", "right"]]) {
            this.stage.append(element("span", `position:absolute;top:8px;${side}:8px;background:#111b;padding:3px 7px;border-radius:4px;pointer-events:none;`, name));
        }
        this.message = element("div", "font-size:11px;line-height:1.4;min-height:16px;", "Connect two IMAGE sequences and queue to compare.");
        this.message.setAttribute("role", "status");
        const controls = element("div", "display:flex;align-items:center;gap:8px;flex-wrap:wrap;flex-shrink:0;");
        this.playButton = element("button", "cursor:pointer;padding:4px 12px;", "Play");
        this.playButton.type = "button";
        this.playButton.disabled = true;
        this.loop = document.createElement("input");
        this.loop.type = "checkbox";
        this.loop.checked = node.properties?.maiVideoCompareLoop ?? true;
        const loopLabel = element("label", "display:flex;align-items:center;gap:3px;cursor:pointer;");
        loopLabel.append(this.loop, document.createTextNode("Loop"));
        this.counter = element("span", "margin-left:auto;font-variant-numeric:tabular-nums;", "0 / 0");
        controls.append(this.playButton, loopLabel, this.counter);
        this.timeline = document.createElement("input");
        this.timeline.type = "range";
        this.timeline.min = "0";
        this.timeline.max = "0";
        this.timeline.step = "1";
        this.timeline.value = "0";
        this.timeline.disabled = true;
        this.timeline.style.cssText = "width:100%;margin:0;flex-shrink:0;";
        this.timeline.setAttribute("aria-label", "Comparison frame");
        const revealLabel = element("label", "display:flex;align-items:center;gap:8px;font-size:11px;flex-shrink:0;", "Reveal A/B");
        this.revealSlider = document.createElement("input");
        this.revealSlider.type = "range";
        this.revealSlider.min = "0";
        this.revealSlider.max = "100";
        this.revealSlider.value = "50";
        this.revealSlider.style.cssText = "flex:1;min-width:50px;margin:0;";
        this.revealSlider.setAttribute("aria-label", "Reveal A/B");
        revealLabel.append(this.revealSlider);
        this.root.append(this.stage, controls, this.timeline, revealLabel, this.message);
        // DOM controls must not drag the workflow canvas or trigger graph hotkeys.
        for (const type of ["pointerdown", "wheel", "keydown"]) {
            this.root.addEventListener(type, (event) => event.stopPropagation());
        }
        this.playButton.onclick = () => this.playing ? this.pause() : this.play();
        this.loop.onchange = () => {
            node.properties ??= {};
            node.properties.maiVideoCompareLoop = this.loop.checked;
        };
        this.timeline.oninput = () => {
            this.pause();
            this.showFrame(Number(this.timeline.value));
        };
        this.revealSlider.oninput = () => this.setReveal(Number(this.revealSlider.value) / 100);
        this.canvas.onpointermove = (event) => {
            const rect = this.canvas.getBoundingClientRect();
            const scale = Math.min(rect.width / this.canvas.width, rect.height / this.canvas.height);
            const width = this.canvas.width * scale;
            const left = rect.left + (rect.width - width) / 2;
            this.setReveal((event.clientX - left) / width);
        };
        this.canvas.onkeydown = (event) => {
            if (event.code === "Space") {
                event.preventDefault();
                this.playing ? this.pause() : this.play();
            }
        };
        this.canvas.addEventListener("webglcontextlost", (event) => {
            event.preventDefault();
            this.pause();
            this.message.textContent = "GPU player context lost; waiting for the browser to restore it.";
        });
        this.canvas.addEventListener("webglcontextrestored", () => {
            this.renderer?.dispose();
            this.renderer = undefined;
            this.draw();
            this.message.textContent = this.note ?? "Connect two IMAGE sequences and queue to compare.";
        });
        this.visibilityHandler = () => { if (document.hidden) this.pause(); };
        document.addEventListener("visibilitychange", this.visibilityHandler);
    }

    setReveal(value) {
        this.reveal = Math.max(0, Math.min(1, value));
        this.revealSlider.value = String(Math.round(this.reveal * 100));
        this.draw();
    }

    setData(data) {
        if (this.disposed || data.subfolder === this.data?.subfolder) return;
        this.pause();
        this.clearCache();
        this.pair = null;
        this.data = data;
        this.index = 0;
        this.timeline.max = String(data.frame_count - 1);
        this.timeline.value = "0";
        this.timeline.disabled = false;
        this.playButton.disabled = false;
        // Fit both images into A's aspect ratio without stretching either side.
        const scale = Math.min(1, 1536 / Math.max(data.width, data.height));
        this.canvas.width = Math.max(1, Math.round(data.width * scale));
        this.canvas.height = Math.max(1, Math.round(data.height * scale));
        const [a, b] = data.source_counts;
        this.note = `${data.fps} FPS · ${data.frame_count} paired frames` +
            (a !== b ? ` · A: ${a}, B: ${b}; ends at shorter sequence` : "");
        this.message.textContent = "Loading comparison…";
        this.draw();
        this.note += ` · ${data.preview_device?.startsWith("cuda") ? "CUDA" : "CPU"} previews` +
            ` · ${this.renderer.gl ? "WebGL player" : "2D player (WebGL unavailable)"}`;
        this.showFrame(0);
    }

    loadPair(index) {
        const key = `${this.data.subfolder}/${index}`;
        if (this.cache.has(key)) {
            const cached = this.cache.get(key);
            this.cache.delete(key);
            this.cache.set(key, cached);
            return cached.promise;
        }
        const data = this.data;
        const cancellations = new Set();
        const promise = Promise.all(["a", "b"].map((side) => new Promise((resolve, reject) => {
            const image = new Image();
            const finish = () => {
                cancellations.delete(cancel);
                image.onload = image.onerror = null;
            };
            const cancel = () => {
                finish();
                image.src = "";
                reject(new Error("Preview load cancelled."));
            };
            cancellations.add(cancel);
            image.onload = () => { finish(); resolve(image); };
            image.onerror = () => {
                finish();
                reject(new Error("Preview unavailable. Change a preview setting and queue again to regenerate frames."));
            };
            const params = new URLSearchParams({
                filename: `${side}_${String(index).padStart(6, "0")}.png`,
                subfolder: data.subfolder, type: "temp",
            });
            image.src = api.apiURL(`/view?${params}`);
        })));
        const cancel = () => { for (const stop of [...cancellations]) stop(); };
        this.cache.set(key, { promise, cancel });
        // Bound decoded image memory for long clips. In-flight frames are also
        // bounded by the sequential playback loop plus three prefetched pairs.
        while (this.cache.size > 12) {
            const oldest = this.cache.keys().next().value;
            this.cache.get(oldest).cancel();
            this.cache.delete(oldest);
        }
        return promise;
    }

    clearCache() {
        for (const cached of this.cache.values()) cached.cancel();
        this.cache.clear();
    }

    async showFrame(index) {
        const request = ++this.request;
        try {
            const pair = await this.loadPair(index);
            if (this.disposed || request !== this.request) return;
            this.pair = pair;
            this.index = index;
            this.timeline.value = String(index);
            this.counter.textContent = `${index + 1} / ${this.data.frame_count} · ${(index / this.data.fps).toFixed(2)}s`;
            this.message.textContent = this.note;
            this.draw();
            for (let offset = 1; offset <= 3; offset++) {
                const next = index + offset;
                if (next < this.data.frame_count) this.loadPair(next).catch(() => {});
            }
        } catch (error) {
            if (this.disposed || request !== this.request) return;
            this.pause();
            this.message.textContent = error.message;
        }
    }

    draw() {
        this.renderer ??= new GpuCompareRenderer(this.canvas);
        if (this.renderer.gl) {
            this.renderer.draw(this.canvas, this.pair, this.reveal);
            return;
        }
        const ctx = this.canvas.getContext("2d");
        const { width, height } = this.canvas;
        ctx.fillStyle = "#08090b";
        ctx.fillRect(0, 0, width, height);
        if (!this.pair) return;
        const fit = (image) => {
            const scale = Math.min(width / image.width, height / image.height);
            const w = image.width * scale, h = image.height * scale;
            ctx.drawImage(image, (width - w) / 2, (height - h) / 2, w, h);
        };
        fit(this.pair[1]);
        const x = width * this.reveal;
        ctx.save();
        ctx.beginPath();
        ctx.rect(0, 0, x, height);
        ctx.clip();
        // Clear B under A, including letterbox regions and transparent pixels.
        ctx.fillRect(0, 0, width, height);
        fit(this.pair[0]);
        ctx.restore();
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = Math.max(2, width / 400);
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
        ctx.fillStyle = "#fff";
        ctx.beginPath();
        ctx.arc(x, height / 2, Math.max(5, width / 100), 0, Math.PI * 2);
        ctx.fill();
    }

    play() {
        if (!this.data || this.disposed || this.playing) return;
        this.playing = true;
        const playbackId = ++this.playbackId;
        this.playButton.textContent = "Pause";
        this.startIndex = this.index >= this.data.frame_count - 1 ? 0 : this.index;
        this.started = performance.now();
        const tick = async (now) => {
            if (!this.playing || this.disposed || playbackId !== this.playbackId) return;
            const elapsed = this.startIndex + Math.floor((now - this.started) * this.data.fps / 1000);
            const ended = !this.loop.checked && elapsed >= this.data.frame_count - 1;
            const index = ended ? this.data.frame_count - 1 : elapsed % this.data.frame_count;
            if (index !== this.index || !this.pair) await this.showFrame(index);
            if (!this.playing || this.disposed || playbackId !== this.playbackId) return;
            if (ended) this.pause();
            else this.raf = requestAnimationFrame(tick);
        };
        this.raf = requestAnimationFrame(tick);
    }

    pause() {
        this.playing = false;
        ++this.playbackId;
        this.playButton.textContent = "Play";
        cancelAnimationFrame(this.raf);
        ++this.request;
    }

    dispose() {
        this.pause();
        this.disposed = true;
        this.clearCache();
        this.pair = null;
        this.renderer?.dispose();
        document.removeEventListener("visibilitychange", this.visibilityHandler);
        this.root.remove();
    }
}

app.registerExtension({
    name: "mAI.VideoCompare",
    onNodeOutputsUpdated(outputs) {
        // Restore cached/history previews when the frontend refreshes outputs.
        const graph = app.rootGraph ?? app.graph;
        for (const [id, message] of Object.entries(outputs)) {
            const data = message?.video_compare?.[0];
            if (data) graph?.getNodeById(id)?.maiVideoCompare?.setData(data);
        }
    },
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MAIVideoCompare") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function (...args) {
            const result = created?.apply(this, args);
            this.maiVideoCompare = new VideoComparePlayer(this);
            const widget = this.addDOMWidget("video_compare_player", "mai_video_compare", this.maiVideoCompare.root, {
                serialize: false, hideOnZoom: false,
                getMinHeight: () => 300, getHeight: () => 340,
            });
            widget.serialize = false;
            widget.serializeValue = () => undefined;
            this.setSize([Math.max(this.size[0], 440), Math.max(this.size[1], 490)]);
            return result;
        };
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message, ...args) {
            const result = executed?.call(this, message, ...args);
            const data = message?.video_compare?.[0];
            if (data) this.maiVideoCompare?.setData(data);
            return result;
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const result = configured?.apply(this, args);
            if (this.maiVideoCompare) {
                this.maiVideoCompare.loop.checked = this.properties?.maiVideoCompareLoop ?? true;
            }
            return result;
        };
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function (...args) {
            this.maiVideoCompare?.dispose();
            return removed?.apply(this, args);
        };
    },
});
