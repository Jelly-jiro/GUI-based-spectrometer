#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GUI_spectrum_v1_2 - copy of GUI_spectrum_v1_1 for further modifications
This file is a direct copy of GUI_spectrum_v1_1.py. Future edits will be
performed against this file (v1_2).
Run: python3 GUI_spectrum_v1_2.py
"""

# The contents are identical to GUI_spectrum_v1_1.py; copied for v1_2 work.

import os
import json
import threading
import tempfile
import csv
import subprocess
import sys

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog

import numpy as np
from picamera2 import Picamera2
import cv2
from PIL import Image, ImageTk

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import time

PROJ_DIR = os.path.dirname(__file__)
ROI_PATH = os.path.join(PROJ_DIR, 'roi.json')

TMP_DIR = tempfile.gettempdir()
PLOT_PATH = os.path.join(TMP_DIR, 'spectrum_wavelength_plot.png')
CSV_PATH = os.path.join(TMP_DIR, 'spectrum_wavelength.csv')
CROP_PATH = os.path.join(TMP_DIR, 'roi_crop_latest.jpg')


def safe_bind(cb, name=None):
    """Return a safe wrapper for a canvas binding callback that prints exceptions.

    Used in the ROI selector where a failure shouldn't crash the whole GUI.
    """
    def _wrapped(event, _cb=cb):
        try:
            return _cb(event)
        except Exception:
            import traceback
            traceback.print_exc()
    return _wrapped

# Debugging disabled in cleaned version
DEBUG = False


class SpectrumGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Spectrum Capture (clean)')
        self.geometry('1000x700')

        topfrm = ttk.Frame(self)
        topfrm.pack(fill='x', padx=8, pady=8)

        self.btn_capture = ttk.Button(topfrm, text='Capture Spectrum', command=self.on_capture)
        self.btn_capture.pack(side='left')
        # Integration time input (seconds)
        self._integration_var = tk.DoubleVar(value=1.0)
        integ_lbl = ttk.Label(topfrm, text='Integration time (s):')
        integ_lbl.pack(side='left', padx=(8,0))
        self.integ_entry = ttk.Entry(topfrm, width=8, textvariable=self._integration_var)
        self.integ_entry.pack(side='left')
        self.btn_roi = ttk.Button(topfrm, text='Select ROI', command=self.run_roi_selector)
        self.btn_roi.pack(side='left', padx=(8,0))
        self.btn_reselect = ttk.Button(topfrm, text='Re-select ROI', command=self.on_roi_reselect)
        self.btn_reselect.pack(side='left', padx=(8,0))
        self.btn_bg = ttk.Button(topfrm, text='Capture Background', command=self.on_bg_capture)
        self.btn_bg.pack(side='left', padx=(8,0))
        self.btn_calib = ttk.Button(topfrm, text='Calibration', command=self.on_calibration)
        self.btn_calib.pack(side='left', padx=(8,0))
        self.btn_save_plot = ttk.Button(topfrm, text='Save Plot', command=self.save_plot_file)
        self.btn_save_plot.pack(side='left', padx=(8,0))
        self.btn_save_crop = ttk.Button(topfrm, text='Save Crop', command=self.save_crop_file)
        self.btn_save_crop.pack(side='left', padx=(8,0))
        self.btn_export = ttk.Button(topfrm, text='Export CSV', command=self.export_csv)
        self.btn_export.pack(side='right')

        self.status = ttk.Label(self, text='Ready')
        self.status.pack(fill='x', padx=8)

        content = ttk.Frame(self)
        content.pack(fill='both', expand=True, padx=8, pady=8)

        self.display_frame = ttk.Frame(content)
        self.display_frame.pack(side='left', fill='both', expand=True)
        try:
            self.display_frame.pack_propagate(False)
        except Exception:
            pass

        # mode selector above the plot: Counts or Counts per second
        mode_frame = ttk.Frame(self.display_frame)
        mode_frame.pack(side='top', fill='x')
        self._y_mode = tk.StringVar(value='counts')
        rb1 = ttk.Radiobutton(mode_frame, text='Counts', variable=self._y_mode, value='counts')
        rb2 = ttk.Radiobutton(mode_frame, text='Counts per second', variable=self._y_mode, value='cps')
        rb1.pack(side='left', padx=(4,4))
        rb2.pack(side='left')
        # allow manual R<->B swap when camera returns BGR
        self._swap_rb = tk.BooleanVar(value=False)
        cb_swap = ttk.Checkbutton(mode_frame, text='Swap R↔B', variable=self._swap_rb)
        cb_swap.pack(side='left', padx=(8,0))
        self.plot_label = ttk.Label(self.display_frame)
        self.plot_label.pack(fill='both', expand=True)
        self.crop_label = ttk.Label(self.display_frame)
        self.crop_label.pack(fill='both', expand=True)

        table_frame = ttk.Frame(content, width=320)
        table_frame.pack(side='right', fill='y')
        # columns: wavelength, raw luminance, correction factor (editable), corrected luminance (auto)
        # allow multiple row selection (extended) so user can select many rows
        self.table = ttk.Treeview(table_frame, columns=('wavelength','L','correction','corrected'), show='headings', selectmode='extended')
        self.table.heading('wavelength', text='Wavelength (nm)')
        self.table.heading('L', text='Luminance')
        self.table.heading('correction', text='Correction factor')
        self.table.heading('corrected', text='Corrected luminance')
        # allow resizing
        self.table.column('wavelength', width=100, anchor='center')
        self.table.column('L', width=100, anchor='e')
        self.table.column('correction', width=120, anchor='e')
        self.table.column('corrected', width=120, anchor='e')
        self.table.pack(fill='both', expand=True)

        self.table.bind('<Control-c>', self.copy_selection)
        # track clicks to know which cell the user focused (for paste start)
        self.table.bind('<Button-1>', self._on_tree_click)
        # paste from clipboard (Excel-style tab/newline blocks)
        self.bind_all('<Control-v>', self._on_paste_clipboard)
        self.bind_all('<Control-V>', self._on_paste_clipboard)
        # allow editing cells by double-click
        self.table.bind('<Double-1>', self._on_tree_double_click)
        # allow Enter key to begin editing the focused cell
        self.table.bind('<Return>', self._edit_focused_cell)
        self.protocol('WM_DELETE_WINDOW', self.on_close)

        # image holders
        self._plot_pil = None
        self._crop_pil = None
        self._plot_img = None
        self._crop_img = None
        self._plot_dirty = False
        self._crop_dirty = False
        self._bg_L = None
        self._bg_ts = None
        self._wl_coeff = None
        self._last_display_size = (0, 0)
        # focused cell tracking: (rowid, col_idx)
        self._tree_focused = (None, None)
        # load persisted correction factors
        self._corrections = {}
        try:
            self._load_saved_corrections()
        except Exception:
            self._corrections = {}

        self.display_frame.bind('<Configure>', self._on_display_resize)
        self.picam2 = None
        self._camera_lock = threading.Lock()
        # track for potential future UI affordances; do not place overlays that
        # intercept mouse events (they broke double-click editing).
        self._cell_focus = None
        try:
            self._load_calibration()
        except Exception:
            pass

    def _show_cell_focus(self, row, col_idx):
        """Display a thin rectangle around the given treeview cell."""
        try:
            if getattr(self, '_cell_focus', None) is None:
                return
            bbox = self.table.bbox(row, f'#{col_idx+1}')
            if not bbox:
                self._hide_cell_focus()
                return
            x,y,w,h = bbox
            # place inside the treeview
            try:
                self._cell_focus.place(in_=self.table, x=x, y=y, width=w, height=h)
            except Exception:
                # fallback: ignore
                pass
        except Exception:
            pass

    def _hide_cell_focus(self):
        try:
            if getattr(self, '_cell_focus', None) is not None:
                try:
                    self._cell_focus.place_forget()
                except Exception:
                    pass
        except Exception:
            pass

    def set_status(self, txt):
        self.status.config(text=txt)
        self.update_idletasks()

    def run_roi_selector(self):
        selector = os.path.join(PROJ_DIR, 'select_roi.py')
        if not os.path.isfile(selector):
            messagebox.showerror('Error', 'select_roi.py is missing')
            return
        self.set_status('ROI selector...')
        subprocess.run([sys.executable, selector])
        self.set_status('ROI selector finished')

    def on_roi_reselect(self):
        self.btn_reselect.config(state='disabled')
        threading.Thread(target=self._capture_frame_for_roi, daemon=True).start()

    def on_bg_capture(self):
        self.btn_bg.config(state='disabled')
        threading.Thread(target=self._capture_background, daemon=True).start()

    def _capture_background(self):
        try:
            if not os.path.isfile(ROI_PATH):
                self.after(0, lambda: messagebox.showinfo('ROI missing', 'No ROI. Please select an ROI first'))
                return
            with open(ROI_PATH,'r') as f:
                roi = json.load(f)
            x,y,w,h = int(roi['x']), int(roi['y']), int(roi['w']), int(roi['h'])

            try:
                self._ensure_camera()
            except Exception as e:
                self.after(0, lambda: messagebox.showerror('Camera init', f'Camera initialization failed: {e}'))
                return

            # integrate frames for the requested integration time
            try:
                integ_target = float(self._integration_var.get())
            except Exception:
                integ_target = 1.0
            if integ_target <= 0:
                integ_target = 1.0

            sum_col = None
            sum_img = None
            count = 0
            start = time.time()
            elapsed = 0.0
            # capture at least one frame and keep capturing until elapsed >= target
            while elapsed < integ_target or count == 0:
                with self._camera_lock:
                    frame = self.picam2.capture_array()
                frame = np.asarray(frame, dtype=np.uint8)
                # optionally swap R and B channels if camera buffer is BGR
                try:
                    if getattr(self, '_swap_rb', None) and self._swap_rb.get():
                        frame = frame[..., ::-1]
                except Exception:
                    pass
                H,W = frame.shape[:2]
                x1,y1 = max(0,x), max(0,y)
                x2,y2 = min(W,x+w), min(H,y+h)
                if x1>=x2 or y1>=y2:
                    self.after(0, lambda: messagebox.showerror('ROI error','ROI is outside image bounds'))
                    return
                crop = frame[y1:y2, x1:x2]
                col_mean = crop.mean(axis=0)
                if sum_col is None:
                    sum_col = np.array(col_mean, dtype=float)
                else:
                    sum_col += col_mean
                if sum_img is None:
                    sum_img = crop.astype(float)
                else:
                    sum_img += crop.astype(float)
                count += 1
                elapsed = time.time() - start

            # compute integrated (summed) luminance across the integration period
            R = sum_col[:,0]; G = sum_col[:,1]; B = sum_col[:,2]
            L = 0.299*R + 0.587*G + 0.114*B
            self._bg_L = np.array(L, dtype=float)
            # store the measured integration duration (seconds)
            self._bg_integration = elapsed if elapsed > 0 else float(count)
            self._bg_ts = time.time()
            # save averaged crop image for visual inspection (SNR improves when frames are averaged)
            try:
                if sum_img is not None and count > 0:
                    avg_img = np.clip(sum_img / float(count), 0, 255).astype(np.uint8)
                    pimg = os.path.join(TMP_DIR, 'spectrum_background_crop.jpg')
                    # Use PIL to save to avoid RGB/BGR channel-order confusion
                    try:
                        Image.fromarray(avg_img).save(pimg)
                    except Exception:
                        # fallback to OpenCV write if PIL fails
                        try:
                            cv2.imwrite(pimg, cv2.cvtColor(avg_img, cv2.COLOR_RGB2BGR))
                        except Exception:
                            pass
            except Exception:
                pass
            try:
                p = os.path.join(TMP_DIR, 'spectrum_background.csv')
                with open(p,'w',newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['pixel','L','integration_s'])
                    for i,val in enumerate(self._bg_L):
                        w.writerow([i, float(val), float(self._bg_integration)])
            except Exception:
                pass
            self.after(0, lambda: self.set_status(f'Background captured ({len(self._bg_L)} points)'))
        finally:
            self.after(0, lambda: self.btn_bg.config(state='normal'))

    def _capture_frame_for_roi(self):
        try:
            try:
                self._ensure_camera()
            except Exception as e:
                self.after(0, lambda: messagebox.showerror('Camera init', f'Camera initialization failed: {e}'))
                return
            with self._camera_lock:
                frame = self.picam2.capture_array()
            arr = np.asarray(frame, dtype=np.uint8)
            try:
                if getattr(self, '_swap_rb', None) and self._swap_rb.get():
                    arr = arr[..., ::-1]
            except Exception:
                pass
            img = Image.fromarray(arr)
            current_roi = None
            try:
                if os.path.isfile(ROI_PATH):
                    with open(ROI_PATH,'r') as f:
                        r = json.load(f)
                        current_roi = (int(r['x']), int(r['y']), int(r['w']), int(r['h']))
            except Exception:
                current_roi = None
            self.after(0, lambda: self._open_roi_editor(img, current_roi))
        finally:
            self.after(0, lambda: self.btn_reselect.config(state='normal'))

    def _open_roi_editor(self, pil_img, current_roi=None):
        win = tk.Toplevel(self)
        win.title('Re-select ROI')

        max_w, max_h = 1200, 800
        img_w, img_h = pil_img.size
        scale = min(1.0, max_w / img_w, max_h / img_h)
        disp_w = int(img_w * scale)
        disp_h = int(img_h * scale)
        disp_img = pil_img.resize((disp_w, disp_h), resample=Image.BILINEAR)
        disp_tk = ImageTk.PhotoImage(disp_img)

        canvas = tk.Canvas(win, width=disp_w, height=disp_h, cursor='cross')
        canvas.pack(side='top', fill='both', expand=True)
        canvas_img = canvas.create_image(0,0,anchor='nw', image=disp_tk)

        rect_id = None
        if current_roi is not None:
            cx,cy,cw,ch = current_roi
            x1 = int(cx * scale); y1 = int(cy * scale)
            x2 = int((cx+cw) * scale); y2 = int((cy+ch) * scale)
            rect_id = canvas.create_rectangle(x1,y1,x2,y2, outline='red', width=2)

        info_frame = ttk.Frame(win)
        info_frame.pack(side='bottom', fill='x')
        ok_btn = ttk.Button(info_frame, text='OK', state='disabled')
        ok_btn.pack(side='right', padx=6, pady=6)
        cancel_btn = ttk.Button(info_frame, text='Cancel')
        cancel_btn.pack(side='right', padx=6, pady=6)

        selection = {'x1':None,'y1':None,'x2':None,'y2':None}

        def on_button_press(event):
            selection['x1'] = max(0, min(disp_w-1, event.x))
            selection['y1'] = max(0, min(disp_h-1, event.y))
            nonlocal rect_id
            if rect_id is not None:
                try:
                    canvas.delete(rect_id)
                except Exception:
                    pass
                rect_id = None
            rect_id = canvas.create_rectangle(selection['x1'], selection['y1'], selection['x1'], selection['y1'], outline='red', width=2)
            selection['x2'] = selection['x1']
            selection['y2'] = selection['y1']
            ok_btn.config(state='disabled')

        def on_move(event):
            if selection['x1'] is None:
                return
            x2 = max(0, min(disp_w-1, event.x))
            y2 = max(0, min(disp_h-1, event.y))
            selection['x2'] = x2; selection['y2'] = y2
            try:
                canvas.coords(rect_id, selection['x1'], selection['y1'], x2, y2)
            except Exception:
                pass

        def on_release(event):
            # finalize selection rectangle and enable OK if area > 0
            try:
                if selection['x1'] is None:
                    return
                x2 = max(0, min(disp_w-1, event.x))
                y2 = max(0, min(disp_h-1, event.y))
                selection['x2'] = x2; selection['y2'] = y2
                if selection['x1'] != selection['x2'] and selection['y1'] != selection['y2']:
                    try:
                        ok_btn.config(state='normal')
                    except Exception:
                        pass
            except Exception:
                import traceback; traceback.print_exc()

        def on_ok():
            # compute ROI in original image coords and save
            try:
                if selection['x1'] is None or selection['x2'] is None:
                    return
                sx = min(selection['x1'], selection['x2'])
                sy = min(selection['y1'], selection['y2'])
                ex = max(selection['x1'], selection['x2'])
                ey = max(selection['y1'], selection['y2'])
                # map back to original image coordinates using scale
                ox = int(max(0, min(img_w-1, round(sx / scale))))
                oy = int(max(0, min(img_h-1, round(sy / scale))))
                ow = int(max(1, min(img_w - ox, round((ex - sx) / scale))))
                oh = int(max(1, min(img_h - oy, round((ey - sy) / scale))))
                d = {'x': ox, 'y': oy, 'w': ow, 'h': oh}
                try:
                    with open(ROI_PATH,'w') as f:
                        json.dump(d, f)
                except Exception:
                    pass
                try:
                    win.destroy()
                except Exception:
                    pass
            except Exception:
                import traceback; traceback.print_exc()

        def on_cancel():
            try:
                win.destroy()
            except Exception:
                pass

        canvas.bind('<ButtonPress-1>', safe_bind(on_button_press, 'press'))
        canvas.bind('<B1-Motion>', safe_bind(on_move, 'move'))
        canvas.bind('<ButtonRelease-1>', safe_bind(on_release, 'release'))
        ok_btn.config(command=on_ok)
        cancel_btn.config(command=on_cancel)

        win._disp_tk = disp_tk
        def _on_close():
            try:
                win.destroy()
            except Exception:
                pass
        win.protocol('WM_DELETE_WINDOW', _on_close)

    def on_capture(self):
        self.btn_capture.config(state='disabled')
        threading.Thread(target=self.capture_and_process, daemon=True).start()

    def capture_and_process(self):
        try:
            if not os.path.isfile(ROI_PATH):
                messagebox.showinfo('ROI missing', 'No ROI. Please select an ROI first')
                self.set_status('ROI missing')
                return
            with open(ROI_PATH,'r') as f:
                roi = json.load(f)
            x,y,w,h = int(roi['x']), int(roi['y']), int(roi['w']), int(roi['h'])
            self.set_status('Capturing...')

            try:
                self._ensure_camera()
            except Exception as e:
                messagebox.showerror('Camera init', f'Camera initialization failed: {e}')
                self.set_status('Camera init failed')
                return

            # integrate frames for the requested integration time
            try:
                integ_target = float(self._integration_var.get())
            except Exception:
                integ_target = 1.0
            if integ_target <= 0:
                integ_target = 1.0

            sum_col = None
            sum_img = None
            count = 0
            start = time.time()
            elapsed = 0.0
            while elapsed < integ_target or count == 0:
                with self._camera_lock:
                    frame = self.picam2.capture_array()
                frame = np.asarray(frame, dtype=np.uint8)
                try:
                    if getattr(self, '_swap_rb', None) and self._swap_rb.get():
                        frame = frame[..., ::-1]
                except Exception:
                    pass
                H,W = frame.shape[:2]
                x1,y1 = max(0,x), max(0,y)
                x2,y2 = min(W,x+w), min(H,y+h)
                if x1>=x2 or y1>=y2:
                    messagebox.showerror('ROI error','ROI is outside image bounds')
                    self.set_status('ROI OOB')
                    return
                crop = frame[y1:y2, x1:x2]
                col_mean = crop.mean(axis=0)
                if sum_col is None:
                    sum_col = np.array(col_mean, dtype=float)
                else:
                    sum_col += col_mean
                if sum_img is None:
                    sum_img = crop.astype(float)
                else:
                    sum_img += crop.astype(float)
                count += 1
                elapsed = time.time() - start

            # compute integrated luminance and measured integration time
            R = sum_col[:,0]; G = sum_col[:,1]; B = sum_col[:,2]
            L_integrated = 0.299*R + 0.587*G + 0.114*B
            actual_integration = elapsed if elapsed > 0 else float(count)

            # save the averaged crop image (improves SNR visually)
            try:
                if sum_img is not None and count > 0:
                    avg_img = np.clip(sum_img / float(count), 0, 255).astype(np.uint8)
                    # Save with PIL to preserve array channel order (avoid RGB<->BGR swap)
                    try:
                        Image.fromarray(avg_img).save(CROP_PATH)
                    except Exception:
                        try:
                            cv2.imwrite(CROP_PATH, cv2.cvtColor(avg_img, cv2.COLOR_RGB2BGR))
                        except Exception:
                            pass
            except Exception:
                pass
            # apply background subtraction if available (background is integrated counts)
            try:
                L = L_integrated.copy()
                if getattr(self, '_bg_L', None) is not None:
                    bg = self._bg_L
                    n = len(L)
                    m = len(bg)
                    if m != n and m > 1:
                        x_bg = np.linspace(0.0, 1.0, m)
                        x_new = np.linspace(0.0, 1.0, n)
                        bg_interp = np.interp(x_new, x_bg, bg)
                    else:
                        bg_interp = bg.copy() if m==n else np.zeros_like(L)
                    # if background was captured with a different integration time,
                    # scale it to the same units (both are integrated counts so
                    # no scaling needed for 'counts' mode; for cps we'll divide later)
                    L = L - bg_interp
                    L = np.maximum(L, 0.0)
            except Exception:
                L = L_integrated.copy()
            n = len(L)
            pixels = np.arange(n)
            if getattr(self, '_wl_coeff', None) is not None:
                a,b = self._wl_coeff
                wavelengths = a * pixels + b
            else:
                wavelengths = 700.0 - pixels*(300.0/max(1,n-1))

            # prepare output values according to mode
            mode = self._y_mode.get() if getattr(self, '_y_mode', None) is not None else 'counts'
            if mode == 'cps':
                # counts per second: divide integrated counts by measured integration time
                L_out = L / max(1e-12, actual_integration)
            else:
                # raw integrated counts
                L_out = L

            # determine correction factors (use persisted corrections when present)
            corrs = []
            for wl in wavelengths:
                key = f'{float(wl):.2f}'
                corr = float(self._corrections.get(key, 1.0)) if getattr(self, '_corrections', None) is not None else 1.0
                corrs.append(corr)

            # compute corrected luminance series
            corrected_series = np.array([float(v) * float(c) for v,c in zip(L_out, corrs)], dtype=float)

            # write CSV including correction factor and corrected luminance
            with open(CSV_PATH,'w',newline='') as f:
                w = csv.writer(f)
                w.writerow(['wavelength_nm','L','correction_factor','corrected_luminance','integration_time_s','mode'])
                for wl,val,corr,corrected in zip(wavelengths, L_out, corrs, corrected_series):
                    w.writerow([float(wl), float(val), float(corr), float(corrected), float(actual_integration), mode])

            fig = Figure(figsize=(8,3))
            ax = fig.add_subplot(111)
            ax.plot(wavelengths, L_out, color='k')
            # Force x-axis to fixed wavelength range: left=400 nm, right=700 nm
            try:
                ax.set_xlim(400.0, 700.0)
            except Exception:
                pass
            ax.set_xlabel('Wavelength (nm)')
            if mode == 'cps':
                ax.set_ylabel('Counts per second')
            else:
                ax.set_ylabel('Counts')
            fig.tight_layout()
            try:
                # plot raw spectrum (black) and corrected spectrum (red)
                ax.plot(wavelengths, L_out, color='k', label='Raw')
                try:
                    ax.plot(wavelengths, corrected_series, color='r', label='Corrected')
                except Exception:
                    pass
                ax.legend()
                canvas = FigureCanvasAgg(fig)
                canvas.print_figure(PLOT_PATH, dpi=150)
            except Exception:
                try:
                    import matplotlib.pyplot as _plt
                    _plt.figure(figsize=(8,3))
                    _plt.plot(wavelengths, L_out, color='k', label='Raw')
                    try:
                        _plt.plot(wavelengths, corrected_series, color='r', label='Corrected')
                        _plt.legend()
                    except Exception:
                        pass
                    # Force x-axis to fixed wavelength range: left=400 nm, right=700 nm
                    try:
                        _plt.gca().set_xlim(400.0, 700.0)
                    except Exception:
                        pass
                    _plt.xlabel('Wavelength (nm)')
                    _plt.ylabel('Average luminance')
                    _plt.tight_layout()
                    _plt.savefig(PLOT_PATH, dpi=150)
                    _plt.close()
                except Exception:
                    pass

            self.after(0, lambda: self.update_ui_with_results(PLOT_PATH, CROP_PATH, CSV_PATH))
            self.set_status('Done')
        except Exception as e:
            messagebox.showerror('Error', str(e))
            self.set_status('Error')
        finally:
            self.after(0, lambda: self.btn_capture.config(state='normal'))

    def _ensure_camera(self):
        if self.picam2 is not None:
            return
        pc = Picamera2()
        try:
            pc.configure(pc.create_preview_configuration({"format":"RGB888","size":(1280,720)}))
        except Exception:
            try:
                pc.configure(pc.create_preview_configuration({"format":"XBGR8888","size":(1280,720)}))
            except Exception:
                pc.configure()
        pc.start()
        self.picam2 = pc

    def _stop_camera(self):
        try:
            if self.picam2 is not None:
                try:
                    self.picam2.stop()
                except Exception:
                    pass
                self.picam2 = None
        except Exception:
            pass

    def _calib_path(self):
        return os.path.join(PROJ_DIR, 'calibration.json')

    def _load_calibration(self):
        p = self._calib_path()
        if os.path.isfile(p):
            try:
                with open(p,'r') as f:
                    d = json.load(f)
                if isinstance(d, list) and len(d) == 2:
                    self._wl_coeff = (float(d[0]), float(d[1]))
            except Exception:
                pass

    def _save_calibration(self):
        p = self._calib_path()
        try:
            with open(p,'w') as f:
                json.dump([float(self._wl_coeff[0]), float(self._wl_coeff[1])], f)
            return True
        except Exception:
            return False

    # --- Image helpers (reduce duplicated code) -----------------------------
    def _safe_open_image(self, path, flip_horizontal=False):
        """Open an image safely and optionally return a horizontally flipped copy.

        Returns a PIL Image or None on failure.
        """
        try:
            img = Image.open(path).convert('RGB')
            if flip_horizontal:
                try:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                except Exception:
                    pass
            return img
        except Exception:
            return None

    def _resize_preserve_aspect(self, img, max_w, max_h, max_upscale=1.5):
        """Resize image to fit within (max_w,max_h) preserving aspect ratio.

        Caps upscaling by max_upscale. Returns a new PIL Image.
        """
        if img is None:
            return None
        try:
            scale_w = float(max_w) / max(1, img.width)
            scale_h = float(max_h) / max(1, img.height)
            scale = min(max_upscale, scale_w, scale_h)
            if scale <= 0:
                scale = 1.0
            new_w = max(1, int(img.width * scale))
            new_h = max(1, int(img.height * scale))
            return img.resize((new_w, new_h), resample=Image.BILINEAR)
        except Exception:
            try:
                return img.copy()
            except Exception:
                return None

    def _resize_exact(self, img, w, h):
        """Resize image to exactly (w,h) without preserving aspect ratio."""
        if img is None:
            return None
        try:
            return img.resize((max(1,int(w)), max(1,int(h))), resample=Image.BILINEAR)
        except Exception:
            try:
                c = img.copy()
                return c.resize((max(1,int(w)), max(1,int(h))))
            except Exception:
                return None

    # --- Table / editing helpers ------------------------------------------------
    def _on_tree_click(self, event):
        """Track which cell (row, col) the user clicked so paste can start there."""
        try:
            row = self.table.identify_row(event.y)
            col = self.table.identify_column(event.x)
            if col and col.startswith('#'):
                try:
                    col_idx = int(col.replace('#','')) - 1
                except Exception:
                    col_idx = None
            else:
                col_idx = None
            self._tree_focused = (row if row != '' else None, col_idx)
            # don't create overlay widgets here (they can intercept events).
            # just update the focused-cell coordinates used by paste/Enter-edit.
            pass
        except Exception:
            self._tree_focused = (None, None)

    def _on_tree_double_click(self, event):
        """Inline edit a cell by placing an Entry over the Treeview cell.

        This provides Excel-like editing: double-click to edit a single cell.
        """
        # delegate to shared inline editor starter
        try:
            row = self.table.identify_row(event.y)
            col = self.table.identify_column(event.x)
            if not row or not col:
                return
            try:
                col_idx = int(col.replace('#','')) - 1
            except Exception:
                return
            self._start_inline_edit(row, col_idx)
        except Exception:
            import traceback
            traceback.print_exc()

    def _on_paste_clipboard(self, event=None):
        """Bind target for Ctrl+V — forward to the paste implementation.

        The actual paste logic is implemented in copy_selection (which accepts
        clipboard contents and writes into the table). We simply call it here.
        """
        try:
            self.copy_selection()
        except Exception:
            import traceback
            traceback.print_exc()
        # prevent default handling
        return 'break'

    def _start_inline_edit(self, row, col_idx):
        """Begin inline editing on the given cell (row id, column index).

        Places an Entry widget over the Treeview cell and commits on Return
        or FocusOut. Cancels on Escape.
        """
        try:
            if row is None:
                return
            if col_idx is None:
                return
            try:
                col_name = self.table['columns'][col_idx]
            except Exception:
                return
            if col_name == 'corrected':
                return

            # get bbox and current value
            try:
                bbox = self.table.bbox(row, f'#{col_idx+1}')
            except Exception:
                bbox = None
            cur = self.table.set(row, col_name)

            if bbox and bbox[2] > 5 and bbox[3] > 5:
                x, y, w, h = bbox
                entry = tk.Entry(self.table)
                entry.insert(0, str(cur))
                entry.select_range(0, 'end')
                entry.focus_set()

                def _commit(evt=None):
                    try:
                        val = entry.get().strip()
                        self.table.set(row, col_name, val)
                        if col_name in ('L', 'correction'):
                            self._recalculate_corrected_for_item(row)
                    except Exception:
                        pass
                    finally:
                        try:
                            entry.destroy()
                        except Exception:
                            pass

                def _cancel(evt=None):
                    try:
                        entry.destroy()
                    except Exception:
                        pass

                entry.bind('<Return>', _commit)
                entry.bind('<FocusOut>', _commit)
                entry.bind('<Escape>', _cancel)
                entry.place(x=x, y=y, width=w, height=h)
            else:
                # fallback to dialog
                new = simpledialog.askstring(f'Edit {col_name}', f'New value for {col_name}:', initialvalue=str(cur), parent=self)
                if new is None:
                    return
                self.table.set(row, col_name, new)
                if col_name in ('L', 'correction'):
                    self._recalculate_corrected_for_item(row)
        except Exception:
            import traceback
            traceback.print_exc()

    def _edit_focused_cell(self, event=None):
        """Start inline edit at the currently focused cell (Enter key)."""
        try:
            row, col_idx = getattr(self, '_tree_focused', (None, None))
            if row is None:
                # if no focused row, use first selected row
                sel = list(self.table.selection())
                if sel:
                    row = sel[0]
            if row is None:
                return 'break'
            if col_idx is None:
                col_idx = 2  # default to correction column
            self._start_inline_edit(row, col_idx)
        except Exception:
            import traceback
            traceback.print_exc()
        return 'break'

    def on_calibration(self):
        if not os.path.isfile(CSV_PATH):
            messagebox.showinfo('No spectrum', 'Please capture a spectrum first')
            return

        pixels = []
        Lvals = []
        try:
            with open(CSV_PATH,'r') as f:
                rdr = csv.reader(f)
                next(rdr,None)
                for i,row in enumerate(rdr):
                    if len(row) >= 2:
                        Lvals.append(float(row[1]))
                    else:
                        Lvals.append(0.0)
            n = len(Lvals)
            pixels = list(range(n))
        except Exception as e:
            messagebox.showerror('CSV read', str(e))
            return

        win = tk.Toplevel(self)
        win.title('Calibration')
        win.geometry('900x500')

        fig = Figure(figsize=(6,4))
        ax = fig.add_subplot(111)
        ax.plot(pixels, Lvals, color='k')
        ax.set_xlabel('Pixel index')
        ax.set_ylabel('Average luminance')

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(side='left', fill='both', expand=True)

        def _setup_canvas_bindings():
            try:
                widge = canvas.get_tk_widget()
                win._calib_widget = widge
                try:
                    win.update_idletasks()
                except Exception:
                    pass

                def _tk_click(ev):
                    try:
                        try:
                            ww = widge.winfo_width(); hh = widge.winfo_height()
                            bbox = ax.get_position()
                            ax_left = int(bbox.x0 * ww)
                            ax_width = int((bbox.x1 - bbox.x0) * ww)
                            x_in_axes = ev.x - ax_left
                            if x_in_axes < 0 or x_in_axes > ax_width:
                                pass
                            else:
                                frac = x_in_axes / max(1, ax_width)
                                xlim = ax.get_xlim()
                                xdata = xlim[0] + frac * (xlim[1] - xlim[0])
                                add_point_at(xdata)
                        except Exception:
                            import traceback; traceback.print_exc()
                    except Exception:
                        import traceback; traceback.print_exc()

                try:
                    widge.bind('<Button-1>', _tk_click)
                except Exception:
                    import traceback; traceback.print_exc()

                try:
                    widge.focus_set()
                except Exception:
                    pass
            except Exception:
                import traceback; traceback.print_exc()

        try:
            win.after(50, _setup_canvas_bindings)
        except Exception:
            _setup_canvas_bindings()

        def _global_click(ev):
            try:
                widge = getattr(win, '_calib_widget', None)
                if widge is None:
                    return
                try:
                    if not widge.winfo_exists():
                        return
                except Exception:
                    return

                try:
                    wx = widge.winfo_rootx(); wy = widge.winfo_rooty()
                    local_x = ev.x_root - wx
                    local_y = ev.y_root - wy
                except Exception:
                    local_x = getattr(ev, 'x', None)
                    local_y = getattr(ev, 'y', None)
                if local_x is None:
                    return

                try:
                    ww = widge.winfo_width(); hh = widge.winfo_height()
                except Exception:
                    return

                if local_x < 0 or local_x > ww or local_y < 0 or local_y > hh:
                    return
                bbox = ax.get_position()
                ax_left = int(bbox.x0 * ww)
                ax_width = int((bbox.x1 - bbox.x0) * ww)
                x_in_axes = local_x - ax_left
                if x_in_axes < 0 or x_in_axes > ax_width:
                    return
                frac = x_in_axes / max(1, ax_width)
                xlim = ax.get_xlim()
                xdata = xlim[0] + frac * (xlim[1] - xlim[0])
                add_point_at(xdata)
            except Exception:
                import traceback; traceback.print_exc()

        try:
            win.bind_all('<Button-1>', _global_click)
        except Exception:
            pass

        win._calib_fig = fig
        win._calib_canvas = canvas

        ctrl = ttk.Frame(win, width=260)
        ctrl.pack(side='right', fill='y')

        pts_frame = ttk.Frame(ctrl)
        pts_frame.pack(fill='both', expand=True, padx=6, pady=6)

        pts_label = ttk.Label(pts_frame, text='Calibration points (pixel -> nm)')
        pts_label.pack()

        entries = []
        vlines = []

        def add_point_at(xpix):
            ip = int(round(xpix))
            if ip < 0 or ip >= len(pixels):
                return
            v = ax.axvline(ip, color='r')
            vlines.append(v)
            row = ttk.Frame(pts_frame)
            lbl = ttk.Label(row, text=f'{ip}')
            lbl.pack(side='left')
            ent = ttk.Entry(row, width=10)
            ent.insert(0, '')
            ent.pack(side='left', padx=6)
            rem = ttk.Button(row, text='Remove', command=lambda r=row,v=v: remove_point(r,v))
            rem.pack(side='left', padx=6)
            row.pack(fill='x', pady=2)
            entries.append((ip, ent, row))
            try:
                canvas.draw()
            except Exception:
                import traceback
                traceback.print_exc()

        def remove_point(row, vline):
            try:
                vlines.remove(vline)
            except Exception:
                pass
            try:
                row.destroy()
            except Exception:
                pass
            try:
                vline.remove()
            except Exception:
                pass
            try:
                for t in list(entries):
                    if len(t) >= 3 and t[2] is row:
                        try:
                            entries.remove(t)
                        except Exception:
                            pass
            except Exception:
                pass
            try:
                canvas.draw()
            except Exception:
                import traceback
                traceback.print_exc()

        def on_click(event):
            try:
                if event.inaxes is None:
                    return
                xpix = event.xdata
                add_point_at(xpix)
            except Exception:
                import traceback
                traceback.print_exc()

        try:
            cid = canvas.mpl_connect('button_press_event', on_click)
            win._calib_cid = cid
        except Exception:
            import traceback
            traceback.print_exc()

        def _on_calib_close():
            try:
                if getattr(win, '_calib_canvas', None) is not None and getattr(win, '_calib_cid', None) is not None:
                    try:
                        win._calib_canvas.mpl_disconnect(win._calib_cid)
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                if getattr(win, '_calib_widget', None) is not None:
                    try:
                        win._calib_widget.unbind('<Button-1>')
                    except Exception:
                        pass
                try:
                    win.unbind_all('<Button-1>')
                except Exception:
                    pass
                try:
                    win._calib_widget = None
                except Exception:
                    pass
            except Exception:
                pass
            try:
                win.destroy()
            except Exception:
                pass
        win.protocol('WM_DELETE_WINDOW', _on_calib_close)

        btn_frame = ttk.Frame(ctrl)
        btn_frame.pack(fill='x', padx=6, pady=6)
        def apply_calib():
            pts = []
            wls = []
            for ip, ent, rw in list(entries):
                try:
                    txt = ent.get().strip()
                except Exception:
                    # widget was removed/destroyed; skip
                    continue
                if txt == '':
                    continue
                try:
                    wl = float(txt)
                except Exception:
                    messagebox.showerror('Value', f'Invalid wavelength: {txt}')
                    return
                pts.append(ip)
                wls.append(wl)
            if len(pts) < 2:
                messagebox.showinfo('Need points', 'At least 2 points are required')
                return
            # linear fit: wl = a*p + b
            a,b = np.polyfit(pts, wls, 1)
            self._wl_coeff = (float(a), float(b))
            saved = self._save_calibration()
            messagebox.showinfo('Calibration', f'Calibration applied. a={a:.6g} b={b:.6g} (saved={saved})')
            win.destroy()

        def reset_calib():
            self._wl_coeff = None
            try:
                p = self._calib_path()
                if os.path.isfile(p):
                    os.remove(p)
            except Exception:
                pass
            messagebox.showinfo('Calibration', 'Calibration reset to default')
            win.destroy()

        apply_btn = ttk.Button(btn_frame, text='Apply', command=apply_calib)
        apply_btn.pack(side='left', padx=6)
        reset_btn = ttk.Button(btn_frame, text='Reset', command=reset_calib)
        reset_btn.pack(side='left', padx=6)

        # hint
        hint = ttk.Label(ctrl, text='Click on the plot to add a calibration point.\nEnter the corresponding wavelength (nm) for each point, then Apply.')
        hint.pack(fill='x', padx=6, pady=6)

    def update_ui_with_results(self, plot_path, crop_path, csv_path):
        try:
            # Use helper to open images (crop is flipped for display only)
            self._plot_pil = self._safe_open_image(plot_path, flip_horizontal=False)
            self._crop_pil = self._safe_open_image(crop_path, flip_horizontal=True)
            # debug output
            if DEBUG:
                try:
                    stp = os.path.getmtime(plot_path) if os.path.isfile(plot_path) else None
                    stc = os.path.getmtime(crop_path) if os.path.isfile(crop_path) else None
                    print(f"[DEBUG] update_ui_with_results: plot_path={plot_path} exists={os.path.isfile(plot_path)} size={self._plot_pil.size} mtime={stp} id={id(self._plot_pil)}")
                    print(f"[DEBUG] update_ui_with_results: crop_path={crop_path} exists={os.path.isfile(crop_path)} size={self._crop_pil.size} mtime={stc} id={id(self._crop_pil)}")
                except Exception:
                    print("[DEBUG] update_ui_with_results: could not read image sizes")

            # try to refresh display; if the display frame isn't mapped yet the
            # immediate refresh may do nothing, so schedule a few retry attempts
            # to ensure images appear.
            # mark images as needing recreation (dirty) so the refresh will
            # update PhotoImage objects even if the display size didn't change
            self._plot_dirty = True
            self._crop_dirty = True
            self._refresh_attempt = 0
            self._schedule_refresh_attempt()
        except Exception as e:
            messagebox.showwarning('Display', str(e))

        for i in self.table.get_children():
            self.table.delete(i)
        try:
            with open(csv_path,'r') as f:
                rdr = csv.reader(f)
                header = next(rdr, None)
                for row in rdr:
                    try:
                        wl = float(row[0]) if len(row) > 0 and row[0] != '' else 0.0
                    except Exception:
                        wl = 0.0
                    try:
                        Lval = float(row[1]) if len(row) > 1 and row[1] != '' else 0.0
                    except Exception:
                        Lval = 0.0
                    # correction factor may be present in column 2 (index 2)
                    try:
                        corr = float(row[2]) if len(row) > 2 and row[2] != '' else 1.0
                    except Exception:
                        corr = 1.0
                    try:
                        corrected = float(row[3]) if len(row) > 3 and row[3] != '' else (Lval * corr)
                    except Exception:
                        corrected = (Lval * corr)
                    # allow persisted correction factors to override or fill-in
                    key = f'{wl:.2f}'
                    if key in self._corrections:
                        corr = float(self._corrections[key])
                    # insert formatted strings
                    self.table.insert('', 'end', values=(f'{wl:.2f}', f'{Lval:.6g}', f'{corr:.6g}', f'{corrected:.6g}'))
        except Exception as e:
            messagebox.showwarning('CSV', str(e))

        # ensure corrected values are up-to-date
        for item in self.table.get_children():
            self._recalculate_corrected_for_item(item)

    def copy_selection(self, event=None):
        # Paste block from clipboard into table starting at focused cell.
        try:
            clip = self.clipboard_get()
        except Exception:
            return
        # split into rows; accept CRLF and LF
        lines = [r for r in clip.splitlines()]
        if not lines:
            return
        # parse each line into cells using tab first, then comma fallback
        grid = []
        for line in lines:
            if '\t' in line:
                cells = line.split('\t')
            else:
                cells = [c.strip() for c in line.split(',')]
            grid.append(cells)

        rows = list(self.table.get_children())
        focused_row, focused_col = self._tree_focused
        sel = list(self.table.selection())
        # If user selected multiple rows and clipboard is single-column, map down selection
        if sel and len(grid) == 1 and len(grid[0]) == 1 and len(sel) > 1:
            vals = [grid[0][0]]
            # apply val or replicate if user provided multiple lines
            for idx,item in enumerate(sel):
                try:
                    col_idx = focused_col if focused_col is not None else 2
                    col_name = self.table['columns'][col_idx]
                    if col_name == 'corrected':
                        continue
                    self.table.set(item, col_name, vals[0])
                    self._recalculate_corrected_for_item(item)
                except Exception:
                    pass
            return

        # otherwise, find start index from focused cell or first selected row
        if focused_row is None:
            if sel:
                focused_row = sel[0]
                focused_col = focused_col if focused_col is not None else 2
            else:
                # nothing to target
                return
        try:
            start_idx = rows.index(focused_row)
        except Exception:
            start_idx = 0
        start_col = focused_col if focused_col is not None else 2

        for r_i, rowvals in enumerate(grid):
            tgt_idx = start_idx + r_i
            if tgt_idx >= len(rows):
                break
            item = rows[tgt_idx]
            for c_j, cell in enumerate(rowvals):
                tgt_col_idx = start_col + c_j
                if tgt_col_idx >= len(self.table['columns']):
                    break
                col_name = self.table['columns'][tgt_col_idx]
                if col_name == 'corrected':
                    continue
                self.table.set(item, col_name, cell)
            self._recalculate_corrected_for_item(item)

    def _recalculate_corrected_for_item(self, item):
        try:
            Ls = self.table.set(item, 'L')
            corr = self.table.set(item, 'correction')
            try:
                Lval = float(str(Ls).strip())
            except Exception:
                Lval = 0.0
            try:
                corrv = float(str(corr).strip())
            except Exception:
                corrv = 1.0
            corrected = Lval * corrv
            self.table.set(item, 'corrected', f'{corrected:.6g}')
            # persist correction factor for this wavelength key
            try:
                wl = float(self.table.set(item, 'wavelength'))
                key = f'{wl:.2f}'
                self._corrections[key] = float(corrv)
                # save immediately
                self._save_saved_corrections()
            except Exception:
                pass
        except Exception:
            pass

    def export_csv(self):
        # export current table contents (including any edited correction factors)
        if not self.table.get_children():
            messagebox.showinfo('No CSV','Please capture a spectrum first')
            return
        p = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[('CSV','*.csv')])
        if p:
            try:
                with open(p,'w',newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['wavelength_nm','L','correction_factor','corrected_luminance'])
                    for item in self.table.get_children():
                        vals = self.table.item(item)['values']
                        # assume ordering: wl, L, correction, corrected
                        w.writerow([float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])])
                messagebox.showinfo('Saved', p)
            except Exception as e:
                messagebox.showerror('Save error', str(e))

    def _corrections_path(self):
        return os.path.join(PROJ_DIR, 'corrections.json')

    def _load_saved_corrections(self):
        p = self._corrections_path()
        if os.path.isfile(p):
            try:
                with open(p,'r') as f:
                    d = json.load(f)
                # ensure numeric values
                self._corrections = {k: float(v) for k,v in d.items()}
            except Exception:
                self._corrections = {}

    def _save_saved_corrections(self):
        p = self._corrections_path()
        try:
            with open(p,'w') as f:
                json.dump(self._corrections, f, indent=2)
        except Exception:
            pass

    def save_plot_file(self):
        if os.path.isfile(PLOT_PATH):
            p = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG','*.png')])
            if p:
                import shutil; shutil.copy(PLOT_PATH,p); messagebox.showinfo('Saved', p)
        else:
            messagebox.showinfo('No plot','Please capture a spectrum first')

    def save_crop_file(self):
        if os.path.isfile(CROP_PATH):
            p = filedialog.asksaveasfilename(defaultextension='.jpg', filetypes=[('JPEG','*.jpg'),('PNG','*.png')])
            if p:
                import shutil; shutil.copy(CROP_PATH,p); messagebox.showinfo('Saved', p)
        else:
            messagebox.showinfo('No crop','Please capture a spectrum first')

    def _on_display_resize(self, event):
        self._refresh_display()

    def _refresh_display(self):
        try:
            from PIL import Image as PILImage

            frame_w = self.display_frame.winfo_width()
            total_h = self.display_frame.winfo_height()
            if DEBUG:
                print(f"[DEBUG] _refresh_display: frame_w={frame_w} total_h={total_h} _plot_pil={'yes' if self._plot_pil is not None else 'no'} _crop_pil={'yes' if self._crop_pil is not None else 'no'} last_size={self._last_display_size}")
            # if not mapped yet, sizes may be very small
            if frame_w <= 10 or total_h <= 10:
                return

            # debounce - avoid repeated resizes when size hasn't changed
            current_size = (frame_w, total_h)
            # if size didn't change, usually nothing to do — but if the
            # PhotoImage objects haven't been created yet we still need to
            # proceed to create them. Only skip when size unchanged AND
            # both images are already present.
            if current_size == self._last_display_size:
                # skip only when both PhotoImage objects already exist AND
                # neither image was marked dirty by a recent capture
                if (self._plot_img is not None and self._crop_img is not None) and not (self._plot_dirty or self._crop_dirty):
                    if DEBUG:
                        try:
                            print('[DEBUG] _refresh_display: skipping since size unchanged and images already set and not dirty')
                        except Exception:
                            pass
                    return
                # else fall through and try to create images now that they may
                # have become available
            self._last_display_size = current_size

            plot_h = int(total_h * 0.55)
            crop_h = max(20, total_h - plot_h - 8)

            # Plot image: limit upscaling to avoid runaway growth
            if self._plot_pil is not None:
                # prefer a helper that preserves aspect ratio and caps upscaling
                img = self._resize_preserve_aspect(self._plot_pil, frame_w, plot_h)
                self._plot_img = ImageTk.PhotoImage(img) if img is not None else None
                self.plot_label.config(image=self._plot_img)
                # we've refreshed the displayed PhotoImage for the plot
                self._plot_dirty = False
                if DEBUG:
                    try:
                        print(f"[DEBUG] plot image created: photo size=({self._plot_img.width()},{self._plot_img.height()})")
                        print(f"[DEBUG] plot_label mapped={self.plot_label.winfo_ismapped()} label_size=({self.plot_label.winfo_width()},{self.plot_label.winfo_height()})")
                    except Exception:
                        pass

            # Crop image: stretch to exactly match the displayed plot width and
            # the available crop height. We intentionally DO NOT preserve the
            # original aspect ratio so the crop will always align with the
            # plot's horizontal extent and follow GUI resizing.
            if self._crop_pil is not None:
                try:
                    target_w = self._plot_img.width() if getattr(self, '_plot_img', None) is not None else frame_w
                except Exception:
                    target_w = frame_w
                target_w = max(1, int(target_w))
                target_h = max(1, int(crop_h))
                img2 = self._resize_exact(self._crop_pil, target_w, target_h)
                self._crop_img = ImageTk.PhotoImage(img2) if img2 is not None else None
                self.crop_label.config(image=self._crop_img)
                # we've refreshed the displayed PhotoImage for the crop
                self._crop_dirty = False
                if DEBUG:
                    try:
                        print(f"[DEBUG] crop image created: photo size=({self._crop_img.width()},{self._crop_img.height()})")
                        print(f"[DEBUG] crop_label mapped={self.crop_label.winfo_ismapped()} label_size=({self.crop_label.winfo_width()},{self.crop_label.winfo_height()})")
                    except Exception:
                        pass
        except Exception:
            # expose exception to terminal for debugging
            import traceback
            traceback.print_exc()

    def _schedule_refresh_attempt(self):
        # attempt to refresh a few times with short delays to handle cases
        # where the widget geometry isn't available immediately after load.
        try:
            # If PhotoImage objects already exist and neither image is marked
            # dirty, there's nothing to do. Otherwise we must proceed to
            # refresh so updated PIL images produce new PhotoImage objects.
            if (getattr(self, '_plot_img', None) is not None and getattr(self, '_crop_img', None) is not None) and not (getattr(self, '_plot_dirty', False) or getattr(self, '_crop_dirty', False)):
                return
        except Exception:
            pass
        self._refresh_display()
        self._refresh_attempt = getattr(self, '_refresh_attempt', 0) + 1
        if self._refresh_attempt < 8:
            self.after(200, self._schedule_refresh_attempt)

    def on_close(self):
        # stop camera if running
        try:
            self._stop_camera()
        finally:
            self.destroy()


if __name__ == '__main__':
    app = SpectrumGUI()
    app.mainloop()