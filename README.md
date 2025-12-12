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
- SMART_RENDER のタイル処理を前提に、EDTの1D処理で使うバッファは `thread_local` で再利用し、行/列ごとのヒープ確保を抑えています。
- 描画側は、SDFの中心値で「境界から十分遠いピクセル」を早期スキップし、境界付近だけ 2x2 の supersample を行います（複雑な輪郭ほど効果が出ます）。
- それでも EDT + 2x2 supersample は重めなので、超高解像度・太いボーダーでは計算量が増えます（入力チェックアウト領域が広がるため）。

## Shape/Text で外側が切れる場合
- Smart Render では、ホストが要求する `output_request.rect` がレイヤー境界でタイトに切られることがあります。
- `Outside` / `Both` で境界外にも描画するため、`PF_RenderOutputFlag_RETURNS_EXTRA_PIXELS` を使って `result_rect/max_result_rect` を拡張しています。
- さらに `PF_OutFlag2_REVEALS_ZERO_ALPHA` を有効にし、AEが「非ゼロαの領域だけ」に `max_result_rect` をクロップしてしまうのを避けています（Shape/Textで重要）。
