# Border

After Effects 用のエフェクトプラグイン「Border」のサンプルです。

## 配置
AeSDK 付属の Examples ツリー内でビルドできるようにしてあります。

- `AfterEffectsSDK/Examples/Template/Border` にこのリポジトリを配置してください

## ビルド（Windows）
- `Win/Border.sln` を Visual Studio で開いて `Release|x64` をビルド

## 実装メモ（精度と見た目）
- 輪郭は SDF（Signed Distance Field）で計算しています。
- 斜めや角で「1px程度ズレて見える」問題を避けるため、距離場は方向バイアスの少ない **EDT（Euclidean Distance Transform）** を使用しています。
- `Inside / Outside` の判定を SDF の符号で“硬く”切らず、境界付近だけマスクを滑らかにして AA の見え方を安定化しています。
- `Threshold=0` のときは、SDF生成に限って 50% alpha（=128）を内部的に使用し、アンチエイリアス素材の「見た目の境界」に合わせやすくしています（UIのThreshold自体はそのままです）。

## パフォーマンス
- EDTの1D処理で使うバッファは `thread_local` で再利用し、行/列ごとのヒープ確保を抑えています。
- SDFの符号付き距離は「反対側クラスまでの距離」から作り、`sqrt` 回数を削減しています（従来の `distToBg - distToFg` 形式より軽い）。
- 描画側は、SDFの中心値で「境界から十分遠いピクセル」を早期スキップし、境界付近だけ 2x2 の supersample を行います（複雑な輪郭ほど効果が出ます）。
- それでも EDT + 2x2 supersample は重めなので、超高解像度・太いボーダーでは計算量が増えます（入力チェックアウト領域が広がるため）。

## Shape/Text で外側が切れる場合
- Shape/Text はレイヤー境界がタイトになりがちで、Smart Render だと境界外ピクセルがクロップされるケースがありました。
- 本プロジェクトでは **Smart Render を使わず**、`PF_Cmd_FRAME_SETUP` で `out_data->width/height/origin` を設定して出力バッファを拡張します（MultiSlicer方式）。
- `PF_OutFlag_I_EXPAND_BUFFER` と `PF_OutFlag2_REVEALS_ZERO_ALPHA` を有効化しています。

## Multi-Frame Rendering（MFR）
- `PF_OutFlag2_SUPPORTS_THREADED_RENDERING` を有効化しており、MFR 対応です。
- もし AE 上で「MFR に最適化されていない」警告が出る場合、古い PiPL フラグがキャッシュされていることがあります。AE を再起動し、古い `.aex` を置き換えた状態で再度確認してください。
