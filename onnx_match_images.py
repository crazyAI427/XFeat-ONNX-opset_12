import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime

from utils import draw_matches


def main():

    # ── file paths ───────────────────────────────────────────────────────────
    img_ref_path   = "assets/ref.png"
    img_curr_path  = "assets/tgt.png"
    ext_model_path = "weights/xfeet/xfeat.onnx"
    match_model_path = "weights/xfeet/matching.onnx"


    # ── load images ──────────────────────────────────────────────────────────
    img_ref   = cv2.imread(img_ref_path)
    img_curr  = cv2.imread(img_curr_path)
    img_ref_rgb  = cv2.cvtColor(img_ref,  cv2.COLOR_BGR2RGB)
    img_curr_rgb = cv2.cvtColor(img_curr, cv2.COLOR_BGR2RGB)

    # to NCHW float32, [0,1]
    ref_tensor  = img_ref_rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    curr_tensor = img_curr_rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    # ── extractor session ────────────────────────────────────────────────────
    ext_sess   = onnxruntime.InferenceSession(ext_model_path, providers=['CPUExecutionProvider'])
    x_name     = ext_sess.get_inputs()[0].name
    y_names    = [out.name for out in ext_sess.get_outputs()]

    ref_feats  = ext_sess.run(y_names, {x_name: ref_tensor})
    curr_feats = ext_sess.run(y_names, {x_name: curr_tensor})
    # ── matcher session ──────────────────────────────────────────────────────
    match_sess = onnxruntime.InferenceSession(match_model_path, providers=['CPUExecutionProvider'])
    m_in_names = [inp.name for inp in match_sess.get_inputs()]
    m_out_names= [out.name for out in match_sess.get_outputs()]

    # assemble inputs ---------------------------------------------------------
    match_inputs = {
        m_in_names[0]: ref_feats[0],   # kpts0
        m_in_names[1]: ref_feats[1],   # feats0
        m_in_names[2]: curr_feats[0],  # kpts1
        m_in_names[3]: curr_feats[1],  # feats1
    }
    outputs = match_sess.run(m_out_names, match_inputs)

    mkpts0_pad, mkpts1_pad = outputs
    valid = np.any(mkpts0_pad != 0, axis=1)
    mkpts0 = mkpts0_pad[valid]
    mkpts1 = mkpts1_pad[valid]
    
    print(mkpts0)

    # ── visualise ────────────────────────────────────────────────────────────
    img_matches = draw_matches(mkpts0, mkpts1, img_ref, img_curr)

    plt.figure(figsize=(20, 20))
    plt.imshow(img_matches[:, :, ::-1])  # BGR→RGB
    plt.axis("off")
    plt.show()

    out_name = os.path.join(os.path.dirname(img_ref_path),
                            f"match_onnx.png")
    cv2.imwrite(out_name, img_matches)


if __name__ == "__main__":
    main()
