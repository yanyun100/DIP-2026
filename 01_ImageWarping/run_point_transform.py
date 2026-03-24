import cv2
import numpy as np
import gradio as gr

# Global variables for storing source and target control points
points_src = []
points_dst = []
image = None

# Reset control points when a new image is uploaded
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()
    points_dst.clear()
    image = img
    return img

# Record clicked points and visualize them on the image
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]

    # Alternate clicks between source and target points
    if len(points_src) == len(points_dst):
        points_src.append([x, y])
    else:
        points_dst.append([x, y])

    # Draw points (blue: source, red: target) and arrows on the image
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # Blue for source
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # Red for target

    # Draw arrows from source to target points
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)

    return marked_image

def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    if len(source_pts) < 2:
        return image

    h, w, c = image.shape
    #生成目标图像的坐标网格 (vx, vy)
    vx, vy = np.meshgrid(np.arange(w), np.arange(h))
    v = np.stack([vx, vy], axis=-1).reshape(-1, 2)  # (N, 2)

    p = target_pts.reshape(-1, 1, 2)  # (n, 1, 2)
    q = source_pts.reshape(-1, 1, 2)  # (n, 1, 2)
    v_expanded = v.reshape(1, -1, 2)  # (1, N, 2)

    #计算权重 (基于变形后的控制点距离)
    dist_sq = np.sum((p - v_expanded) ** 2, axis=2) + eps
    w_i = 1.0 / (dist_sq ** alpha)
    w_sum = np.sum(w_i, axis=0, keepdims=True)
    w_i /= w_sum

    #计算加权质心
    p_star = np.sum(w_i[:, :, np.newaxis] * p, axis=0)
    q_star = np.sum(w_i[:, :, np.newaxis] * q, axis=0)

    #去中心化
    hat_p = p - p_star
    hat_q = q - q_star

    #计算刚性变换
    hat_p_perp = np.stack([-hat_p[:, :, 1], hat_p[:, :, 0]], axis=2)
    hat_q_perp = np.stack([-hat_q[:, :, 1], hat_q[:, :, 0]], axis=2)
    
    vp_star = v - p_star
    vp_star_norm = np.sqrt(np.sum(vp_star**2, axis=1))

    fr_x = np.zeros(len(v))
    fr_y = np.zeros(len(v))

    for i in range(len(p)):
        s1 = hat_p[i, :, 0] * vp_star[:, 0] + hat_p[i, :, 1] * vp_star[:, 1]
        s2 = hat_p_perp[i, :, 0] * vp_star[:, 0] + hat_p_perp[i, :, 1] * vp_star[:, 1]
        
        fr_x += w_i[i, :] * (hat_q[i, :, 0] * s1 + hat_q_perp[i, :, 0] * s2)
        fr_y += w_i[i, :] * (hat_q[i, :, 1] * s1 + hat_q_perp[i, :, 1] * s2)

    fr_norm = np.sqrt(fr_x**2 + fr_y**2) + eps

    #计算目标点在原图中的映射坐标
    map_x = vp_star_norm * (fr_x / fr_norm) + q_star[:, 0]
    map_y = vp_star_norm * (fr_y / fr_norm) + q_star[:, 1]

    #重采样
    map_x = map_x.reshape(h, w).astype(np.float32)
    map_y = map_y.reshape(h, w).astype(np.float32)

    #此时 cv2.remap 会根据 map_x/y 从原图中抓取像素填入目标图相应位置
    warped_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return warped_image

def run_warping():
    global points_src, points_dst, image

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# Clear all selected points
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image

# Build Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Upload Image", interactive=True, width=800)
            point_select = gr.Image(label="Click to Select Source and Target Points", interactive=True, width=800)

        with gr.Column():
            result_image = gr.Image(label="Warped Result", width=800)

    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")

    input_image.upload(upload_image, input_image, point_select)
    point_select.select(record_points, None, point_select)
    run_button.click(run_warping, None, result_image)
    clear_button.click(clear_points, None, point_select)

demo.launch()