# Border

After Effects 用のエフェクトプラグイン「Border」のサンプルです。

## 配置
AeSDK 付属の Examples ツリー内でビルドできるようにしてあります。

- `AfterEffectsSDK/Examples/Template/Border` にこのリポジトリを配置してください

## ビルド（Windows）
- `Win/Border.sln` を Visual Studio で開いて `Release|x64` をビルド

## 動作確認（AA）
- コンポのビット深度を `8bpc` / `16bpc` で切り替えて、同じ素材・同じ設定でエッジの階段状/段差が減っているか確認してください。
- 垂直/水平に近い箇所・曲率が大きい箇所を 400%〜800% で拡大すると差が見えやすいです。

## 実装メモ（精度と見た目）
- 輪郭は SDF（Signed Distance Field）で計算しています。
- 斜めや角での方向バイアス（距離の量子化でAAが“平べったく”見える等）を避けるため、**境界ピクセル集合** に対する **EDT（Euclidean Distance Transform）** を使い、距離をピクセル中心ではなく「境界（alpha==threshold）」に近づけています。
- 境界ピクセルでは Sobel 勾配から `alpha==threshold` の交点を推定し、**サブピクセル位置** への距離としてSDFを補正します（Plateau削減）。
- `Inside / Outside` の判定を SDF の符号で“硬く”切らず、境界付近だけマスクを滑らかにして AA の見え方を安定化しています。
- `Threshold=0` のときは、SDF生成に限って 50% alpha（=128）を内部的に使用し、アンチエイリアス素材の「見た目の境界」に合わせやすくしています（UIのThreshold自体はそのままです）。
- 8bpc 出力では、量子化でAAが段差状に見えやすいので、**軽量ディザ** を入れて見た目の滑らかさを近づけています。

## パフォーマンス
- EDTの1D処理で使うバッファは `thread_local` で再利用し、行/列ごとのヒープ確保を抑えています。
- SDFは「境界ピクセルまでの距離」をEDTで求め、描画に必要な距離の範囲を安定して評価します。
- Shape/Text のように「画面の一部にだけ形がある」ケースでは、SDF/描画を境界近傍のROIに限定して計算量を削減します（拡張バッファ全体に対して EDT を回さない）。
- `Show line only` 以外では、出力初期化を「左右上下の余白帯だけ memset + 本体は memcpy」にしてメモリ帯域を削減します。
- 描画側は、SDFの中心値で「境界から十分遠いピクセル」を早期スキップし、境界付近だけ 2x2 の supersample を行います（複雑な輪郭ほど効果が出ます）。
- それでも EDT + 2x2 supersample は重めなので、超高解像度・太いボーダーでは計算量が増えます（入力チェックアウト領域が広がるため）。
- さらに詰める場合、内部並列（EDTの行/列パス、描画ループ）を **任意で** 有効化できます。環境変数 `BORDER_THREADS` でスレッド数を指定してください（例: `BORDER_THREADS=4`）。デフォルトは **内部1スレッド** です（AEのMFRと過剰並列になって不安定化するのを避けるため）。上限は8です。

## Shape/Text で外側が切れる場合
- Shape/Text はレイヤー境界がタイトになりがちで、Smart Render だと境界外ピクセルがクロップされるケースがありました。
- 本プロジェクトでは **Smart Render を使わず**、`PF_Cmd_FRAME_SETUP` で `out_data->width/height/origin` を設定して出力バッファを拡張します（MultiSlicer方式）。
- `PF_OutFlag_I_EXPAND_BUFFER` と `PF_OutFlag2_REVEALS_ZERO_ALPHA` を有効化しています。

## Multi-Frame Rendering（MFR）
- `PF_OutFlag2_SUPPORTS_THREADED_RENDERING` を有効化しており、MFR 対応です。
- もし AE 上で「MFR に最適化されていない」警告が出る場合、古い PiPL フラグがキャッシュされていることがあります。AE を再起動し、古い `.aex` を置き換えた状態で再度確認してください。
