## Taken from: https://github.com/3288103265/SegGAN/blob/e2abeeb144be7fdfa36d2b28da48644107ce74c6/utils.py#L139
## visualize masks with more than one channels.
def one_hot_to_rgb(layout_pred, colors, num_classes):
    one_hot = layout_pred[:, :num_classes, :, :]
    one_hot_3d = torch.einsum('abcd,be->aecd', [one_hot, colors])
    one_hot_3d *= (255.0 / one_hot_3d.max())
    return one_hot_3d


def one_hot_3d_to_rgb(layout_3d, objs, obj_to_img,  colors):
    """Convert 3-dim layout to RGB mode for visualization.

    Args:
        layout_3d (Tensor[float]): GT layout, O*H*W.
        obj_to_img (tensor[long]): Specify which image H*W belongs to. O*1
        objs (tensor[long]): labels, O*1
        colors (tensor[float]): O*3 range in (0,256)
    Return:
        res(tensor): N*3*H*W, colorized layouy.
    """
    objs = objs.long()
    obj_to_img = obj_to_img.long()
    layout_3d = layout_3d.float()
    split_group = list(Counter(obj_to_img.tolist()).values())
    # split objs and layout according to obj_to_img, so that each split is belonging to an img.
    layouts_splits = layout_3d.split(split_group)
    objs_splits = objs.split(split_group)
    res = []
    for layout, obj in zip(layouts_splits, objs_splits):
        color = colors[obj]
        layout_rgb = torch.einsum("ohw,oc->chw", [layout, color])
        res.append(layout_rgb)

    res = torch.stack(res, dim=0)
    res *= (255.0 / res.max())
    return res

if __name__ == '__main__':
    layout_3d = torch.rand(23,32,32)
    objs = torch.rand(23)
    obj_to_img = torch.tensor([0]*7+[1]*7+[2]*7+[3]*2)
    colors = torch.rand(23,3)
    
    out = one_hot_3d_to_rgb(layout_3d, objs, obj_to_img, colors)
    # print(out.shape)
    assert out.shape == (4,3,32,32)
    print('Test done!')

