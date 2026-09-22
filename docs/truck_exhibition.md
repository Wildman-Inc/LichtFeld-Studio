# truck 学習の展示用ランチャー

`scripts/start_truck_exhibition.bat` をダブルクリックすると、LichtFeld Studio のウィンドウが開き、truck データセットの読み込み後に学習が自動で始まります。学習の進行とモデルを GUI で表示し、学習完了後もウィンドウは開いたままになります。停止・一時停止・再開は LFS の学習パネルから操作できます。

Windows と PowerShell 7、およびビルド済み LFS が必要です。標準では `build-rocm10/Release/LichtFeld-Studio.exe` と、現在のユーザーの `Downloads/tandt_db/tandt/truck` を使用します。`LFS_EXECUTABLE` が設定されている場合は、その実行ファイルを優先します。

展示用の既定値は MRNF、30,000 iteration、最大 100 万 Gaussian、画像を 1/2 に縮小です。起動ごとに「ドキュメント」の `LichtFeld-Exhibition/truck-日時-識別子` を作り、学習結果、`lichtfeld.log`、起動引数を記録した `launch.json` を保存します。

PowerShell から設定を変更できます。

```powershell
# 短い動作確認
./scripts/start_truck_exhibition.ps1 -Iterations 2000 -MaxCap 300000

# データ・実行ファイル・保存先を指定
./scripts/start_truck_exhibition.ps1 `
  -Dataset 'D:/datasets/truck' `
  -Executable 'C:/Dev/LichtFeld-Studio/build-rocm10/Release/LichtFeld-Studio.exe' `
  -OutputRoot 'D:/exhibition-results' `
  -Iterations 30000 -MaxCap 1000000 -ResizeFactor 1

# 起動内容とパスを検証するだけ（GUI 起動・ファイル作成なし）
./scripts/start_truck_exhibition.ps1 -DryRun
```

`-Strategy` は `mrnf` / `mcmc` / `igs+`、`-ResizeFactor` は `1` / `2` / `4` / `8` を指定できます。`-Wait` を付けるとスクリプトは GUI が閉じるまで待機します。通常は起動後にスクリプトが終了しても LFS は動作を続けます。

LFS の公開 CLI `--train` で自動開始します。MCP による追加操作は不要です。MCP ポートはこの起動だけ `45678` に変更し、使用中の場合は `-McpPort 45679` のように変更してください。保存済みの設定は書き換えません。MCP が有効な環境では、表示された endpoint を initialize し、resources/list、tools/list、`lichtfeld://runtime/catalog` と状態リソースを取得した後、`training.main` を `runtime_job_describe` / `runtime_job_wait` で監視できます。

起動に失敗した場合は、コンソールのメッセージと今回の出力先の `lichtfeld.log` を確認してください。既存の LFS は閉じないため、性能測定や別の学習中は、その処理の終了後に展示を起動してください。
