import { app } from "../../../scripts/app.js";

const hiddenStates = new WeakMap();
const boundControls = new WeakSet();
const controllers = new Set([
    "use_mask_max_size", "preresize", "preresize_mode",
    "extend_for_outpainting", "output_resize_to_target_size", "stitch_mask_mode",
]);

function isConnected(node, name) {
    return node.inputs?.some((input) =>
        (input.name === name || input.widget?.name === name) && input.link != null,
    );
}

function updateVisibility(node) {
    const value = (name) => node.widgets?.find((widget) => widget.name === name)?.value;
    // Linked controls can change at execution time: expose both sets of options.
    const boundedLinked = isConnected(node, "use_mask_max_size");
    const bounded = value("use_mask_max_size") === true && !boundedLinked;
    const enabled = (name) => value(name) === true || isConnected(node, name);
    const pre = !bounded && enabled("preresize");
    const preMode = value("preresize_mode") ?? "";
    const preModeLinked = isConnected(node, "preresize_mode");
    const visible = {
        max_crop_width: bounded || boundedLinked,
        max_crop_height: bounded || boundedLinked,
        crop_multiple_of: bounded || boundedLinked,
        preresize: !bounded,
        preresize_mode: pre,
        preresize_min_width: pre && (preMode.includes("minimum") || preModeLinked),
        preresize_min_height: pre && (preMode.includes("minimum") || preModeLinked),
        preresize_max_width: pre && (preMode.includes("maximum") || preModeLinked),
        preresize_max_height: pre && (preMode.includes("maximum") || preModeLinked),
        extend_for_outpainting: !bounded,
        extend_up_factor: !bounded && enabled("extend_for_outpainting"),
        extend_down_factor: !bounded && enabled("extend_for_outpainting"),
        extend_left_factor: !bounded && enabled("extend_for_outpainting"),
        extend_right_factor: !bounded && enabled("extend_for_outpainting"),
        context_from_mask_extend_factor: !bounded,
        output_resize_to_target_size: !bounded,
        output_target_width: !bounded && enabled("output_resize_to_target_size"),
        output_target_height: !bounded && enabled("output_resize_to_target_size"),
        output_padding: !bounded,
        stitch_mask_expand_pixels: value("stitch_mask_mode") === "extended mask"
            || isConnected(node, "stitch_mask_mode"),
    };
    for (const widget of node.widgets ?? []) {
        if (!(widget.name in visible)) continue;
        // Never change sockets, serialization, values, or converted widget types.
        const converted = widget.type?.startsWith("converted-widget");
        const hide = !visible[widget.name] && !converted && !isConnected(node, widget.name);
        if (hide && !hiddenStates.has(widget)) {
            hiddenStates.set(widget, widget.hidden);
            widget.hidden = true;
        } else if (!hide && hiddenStates.has(widget)) {
            widget.hidden = hiddenStates.get(widget);
            hiddenStates.delete(widget);
        }
    }
    const size = node.computeSize();
    node.setSize([Math.max(node.size[0], size[0]), size[1]]);
    node.setDirtyCanvas(true, true);
}

function bindVisibility(node) {
    for (const widget of node.widgets ?? []) {
        if (!controllers.has(widget.name) || boundControls.has(widget)) continue;
        boundControls.add(widget);
        const callback = widget.callback;
        widget.callback = function (...args) {
            const result = callback?.apply(this, args);
            updateVisibility(node);
            return result;
        };
    }
    updateVisibility(node);
}

app.registerExtension({
    name: "mAI.InpaintCropSeparateStitchMask.Visibility",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "MAIInpaintCropSeparateStitchMask") return;
        for (const hook of ["onNodeCreated", "onConfigure", "onConnectionsChange"]) {
            const original = nodeType.prototype[hook];
            nodeType.prototype[hook] = function (...args) {
                const result = original?.apply(this, args);
                queueMicrotask(() => bindVisibility(this));
                return result;
            };
        }
    },
});
