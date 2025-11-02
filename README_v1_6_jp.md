# GUI_spectrum_v1_6 — README (v1.6)

このドキュメントは `GUI_spectrum_v1_6.py` の使用方法とセットアップ手順をまとめたものです。

## 目的
- カメラで ROI（領域）を切り出し、クロップ領域からスペクトル（輝度分布）を算出・表示・保存します。
- プロット（波長軸）は固定で 400 nm 〜 700 nm に設定されています（v1_6 の仕様）。

---

## 必要なソフトウェア／ライブラリ
以下は本 GUI を動作させるために一般的に必要となるパッケージの一覧とインストール方法です。
（Raspberry Pi などの Linux 上で動作することを想定しています）

必須 Python パッケージ（pip）
- numpy
- pillow
- matplotlib
- opencv-python
- picamera2 （カメラ制御用）

システムパッケージ（GUI/カメラ）
- python3-tk（Tkinter GUI）
- libcamera / picamera2 の実行環境（Raspberry Pi OS では libcamera と Picamera2 が必要）

注意: 環境によっては `picamera2` が apt パッケージ経由で提供されることがあります。

### 推奨インストール手順（例）
1. システムパッケージのインストール（Debian / Raspberry Pi OS 系）:

```bash
sudo apt update
sudo apt install -y python3-pip python3-tk libcamera-apps
# (Raspberry Pi OS の場合) picamera2 を apt 経由で提供していることがあるので、必要に応じて:
sudo apt install -y python3-picamera2
```

2. Python パッケージのインストール（pip）:

```bash
python3 -m pip install --user numpy pillow matplotlib opencv-python
# picamera2 が pip で入手可能であれば:
python3 -m pip install --user picamera2
```

3. 注意事項:
- `opencv-python` は GUI（imshow 等）を必要としない場合は `opencv-python-headless` を使う選択肢もありますが、本プロジェクトは画像処理だけで GUI 描画は Pillow/Tkinter を使用しているため `opencv-python` でも動きます。
- Picamera2 は Raspberry Pi 固有のパッケージやカーネルモジュールに依存します。エラーが出る場合は Raspberry Pi のドキュメント（libcamera / Picamera2）や OS のパッケージ管理を先に確認してください。

---

## 起動方法
プロジェクトのディレクトリで次を実行します:

```bash
cd /home/pi2172/Documents/python/spectrometer
python3 GUI_spectrum_v1_6.py
```

- `python3 -m py_compile GUI_spectrum_v1_6.py` で構文チェックができます。
- GUI を実行する端末にエラーが出た場合、そのログ（ターミナル出力）を保存しておくとデバッグが容易です。

---

## 一通りの使用手順（推奨順序）
1. カメラを物理的にセットアップし、OS 側でカメラが有効になっていることを確認します（Raspberry Pi なら `raspi-config` など）。
2. GUI を起動します。
3. 「Select ROI」または「Re-select ROI」をクリックして ROI（解析領域）を選択します。
   - ROI ダイアログが開き、表示画像上でドラッグして矩形領域を選びます。
   - OK を押すと ROI が保存されます（`roi.json` に保存されます）。
4. 必要なら「Capture Background」をクリックして背景（ダーク/ベースライン）を取得します。
   - 取得した背景は取得時点の積分時間で平均化され、後続のスペクトル取得時に差し引かれます。
5. スペクトルを取得するには「Capture Spectrum」ボタンを押します。
   - 指定の積分時間（画面上の Integration time (s)）に従ってフレームを積分します。
   - 取得後、波長 (400–700 nm) に固定したプロットが生成され、クロップ画像と共に GUI に表示されます。
6. 必要に応じて:
   - 「Save Plot」: 現在のプロット（PNG）を保存。
   - 「Save Crop」: 平均化されたクロップ画像を保存。
   - 「Export CSV」: 波長・L・補正係数・補正後輝度などを含む CSV を保存。
7. 色がおかしい（赤が青に見える等）場合は、Mode 領域にある「Swap R↔B」チェックボックスを切り替えて試してください（表示と内部処理で RGB/BGR の入れ替えを行います）。

---

## 各ボタンの機能（GUI 上の表記と説明）
- Capture Spectrum
  - ROI のクロップ領域からスペクトルを取得します。Integration time (s) で指定した時間分フレームを統合し、プロット・CSV・クロップ画像を更新します。

- Select ROI
  - 初回の ROI 選択に使用します。カメラから取得したフレームのプレビューを表示し、マウスで矩形をドラッグして ROI を定義します。

- Re-select ROI
  - 既に ROI が保存されている場合に再設定します。ROI ダイアログで新たに領域を選択し、保存します。

- Capture Background
  - 指定時間分フレームを統合して背景（bg）として保存します。スペクトル取得時にこの背景を差し引くことができます。

- Calibration
  - 現在の CSV（最後に取得したスペクトル）からピクセル→波長のキャリブレーションを行います。プロット上で少なくとも 2 点を選び、それらに対応する波長を入力して線形フィッティングを行います。

- Save Plot
  - 現在のプロットを PNG として保存します（保存先はダイアログで指定）。

- Save Crop
  - 最後に取得した平均化クロップ画像を保存します（JPEG/PNG を選択可能）。

- Export CSV
  - テーブル（波長、L、補正係数、補正後輝度）を CSV としてエクスポートします。

- Swap R↔B (チェックボックス)
  - カメラのカラーチャネル順が BGR で返ってきている場合、表示および内部処理のために R と B を入れ替えます。色がおかしいと感じたらオンにして確認してください。

---

## ファイル出力
- 一時プロット: /tmp/spectrum_wavelength_plot.png
- 出力 CSV: /tmp/spectrum_wavelength.csv
- 平均化クロップ: /tmp/roi_crop_latest.jpg
- ROI 設定: `roi.json`（プロジェクトディレクトリ）

（パスはコード内の定数 `PLOT_PATH`, `CSV_PATH`, `CROP_PATH`, `ROI_PATH` を参照）

---

## トラブルシューティング
- 起動時に `IndentationError` や `SyntaxError` が出た場合
  - ファイルが編集途中でインデントが崩れている可能性があります。最新版の `GUI_spectrum_v1_6.py` を確認してもらうか、私にエラートレースを送ってください。

- カメラ初期化エラー（`Camera initialization failed`）
  - libcamera/Picamera2 の環境が正しく入っているか確認してください。
  - Raspberry Pi では `raspi-config` でカメラインターフェースが有効になっているかを確認してください。

- 赤が青に見える等の色逆転
  - Mode 領域の "Swap R↔B" を切り替えて確認してください。永続設定や自動検出を希望する場合は追加実装できます。

---

## 開発者向けメモ
- 波長軸は v1_6 で 400–700 nm に固定されています。必要ならコード内で `ax.set_xlim(...)` の値を変更してください。
- 画像表示は Pillow (PIL.ImageTk) を経由して Tkinter に渡しています。Matplotlib は一時 PNG に描画してから読み込む方式です。

---

## 変更履歴（簡易）
- v1_6: v1_5 の安定版をコピーし、UI 文言を英語化。プロット x 軸を 400–700 nm に固定。クロップ表示をプロット幅に合わせる変更、R/B swap の UI 追加など。

---

必要があれば、この README を `README.md` に反映したり、英語版を作成したり、インストール手順を OS ごとに詳細化します。どれを次にやりたいか教えてください。
