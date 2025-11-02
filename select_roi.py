#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive ROI selector for Raspberry Pi camera.

- キャプチャしたフレームを表示し、マウスで矩形を選択します。
- 選択を `roi.json` に保存します（形式: {"x":..,"y":..,"w":..,"h":..}）。

使い方:
    python3 select_roi.py

操作:
- マウスでドラッグして矩形を作ります。
- Enter/Space で確定、Esc でキャンセル（もう一度撮り直し）
- ウィンドウを閉じると終了します。

注: GUI（ディスプレイ/VNC）が必要です。
"""

import json
import os
import cv2
import numpy as np
from picamera2 import Picamera2

OUT_ROI = os.path.join(os.path.dirname(__file__), 'roi.json')
OUT_IMG = '/tmp/roi_selection.jpg'


def capture_frame(size=(1280, 720)):
    picam2 = Picamera2()
    try:
        picam2.configure(picam2.create_preview_configuration({"format": "RGB888", "size": size}))
    except Exception:
        try:
            picam2.configure(picam2.create_preview_configuration({"format": "XBGR8888", "size": size}))
        except Exception:
            picam2.configure()
    picam2.start()
    frame = picam2.capture_array()
    picam2.stop()
    return np.asarray(frame, dtype=np.uint8)


def main():
    print('カメラからフレームを取得します...')
    frame = capture_frame()
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(OUT_IMG, bgr)

    while True:
        print('ウィンドウが開きます。マウスで矩形を選んで Enter/Space で確定、Esc でキャンセル/再撮影、q で終了。')
        winname = 'Select ROI - press Enter to confirm'
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        cv2.imshow(winname, bgr)
        # selectROI returns (x,y,w,h)
        roi = cv2.selectROI(winname, bgr, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(winname)
        x, y, w, h = roi
        if w == 0 or h == 0:
            # user cancelled selection
            print('矩形が選択されませんでした。再撮影しますか？ (y/n)')
            choice = input().strip().lower()
            if choice == 'y' or choice == '':
                frame = capture_frame()
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(OUT_IMG, bgr)
                continue
            else:
                print('終了します。')
                return

        # show annotated image and confirm
        annot = bgr.copy()
        cv2.rectangle(annot, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('ROI Preview - press any key', annot)
        k = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if k == 27:  # ESC -> retake
            print('再撮影します...')
            frame = capture_frame()
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(OUT_IMG, bgr)
            continue
        # save roi
        roi_dict = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
        with open(OUT_ROI, 'w') as f:
            json.dump(roi_dict, f, indent=2)
        print(f'ROI saved to {OUT_ROI}:', roi_dict)
        # also save preview image
        cv2.imwrite(OUT_IMG, annot)
        print('Annotated preview saved to', OUT_IMG)
        break


if __name__ == '__main__':
    main()
