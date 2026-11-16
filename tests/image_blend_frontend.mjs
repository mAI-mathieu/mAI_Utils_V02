import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

let extension;
globalThis.testApp = { registerExtension(value) { extension = value; } };
const source = await readFile(new URL("../web/js/image_blend.js", import.meta.url), "utf8");
await import(`data:text/javascript;base64,${Buffer.from(source.replace(
    'import { app } from "../../../scripts/app.js";',
    "const app = globalThis.testApp;",
)).toString("base64")}`);
delete globalThis.testApp;

class Node {
    widgets = [
        { name: "blend_mode", value: "normal" },
        { name: "blend_percentage", value: 50 },
        { name: "blend_list", value: "", hidden: true },
        { name: "detail_scale", value: 2 },
    ];

    onConfigure() {
        return "configured";
    }
}

function configure(values, inputs = []) {
    const node = new Node();
    // LiteGraph assigns serialized widget values before it calls onConfigure.
    values.forEach((value, index) => { node.widgets[index].value = value; });
    assert.equal(node.onConfigure({ widgets_values: values, inputs }), "configured");
    return node.widgets;
}

const original = Node.prototype.onConfigure;
extension.beforeRegisterNodeDef(Node, { name: "OtherNode" });
assert.equal(Node.prototype.onConfigure, original);
extension.beforeRegisterNodeDef(Node, { name: "MAIImageBlend" });

for (const [values, inputs, expected, legacyHidden] of [
    [[], [], ["normal", 50, "", 2], true],
    [["25.75%,\n"], [], ["normal", 25.75, "", 2], true],
    [["0%, 50%, 100%"], [], ["normal", 50, "0%, 50%, 100%", 2], false],
    [[null], [{ name: "blend_list", link: 8 }], ["normal", 50, "", 2], false],
    [["multiply", 33.33, ""], [], ["multiply", 33.33, "", 2], true],
    [["normal", 50, "0%, 50%, 100%"], [], ["normal", 50, "0%, 50%, 100%", 2], false],
    [["high_frequency", 25, "", 3.1], [], ["high_frequency", 25, "", 3.1], true],
]) {
    const widgets = configure(values, inputs);
    assert.deepEqual(widgets.map(({ value }) => value), expected);
    assert.equal(widgets[2].hidden, legacyHidden);
}

console.log("Blend dropdown and float configuration preserved legacy workflow settings.");
