# galaxy-morphology  
最近の「オープンサイエンス/オープンデータ」の潮流は、「一般人」にとっても科学を身近なものにさせてくれる。そこで、具体的にどんなことができるのか、挑戦してみることにした。  
このプロジェクトでは、[Galaxy Zoo](https://data.galaxyzoo.org/)と[SDSS(The Sloan Digital Sky Survey)](http://www.sdss.org/)から公開されているデータを使って、銀河形態の（ごく初歩的な）自動分類を深層学習によって実装している。  
### 環境要件  
- Ubuntu 16.04 LTS (amd64)  
- CUDA Toolkit 9.1  
- cuDNN v7.1.3  
- Python 3.6 (Anaconda 5.1)  
- TensorFlow 1.9.0 (GPUサポート)  
- Keras 2.2.0  
- 動作確認ハード
  - Dell OptiPlex 9010 MT (電源部増強)  
  - CPU Intel® Core™ i7-3770 @ 3.40GHz × 8
  - メモリ 16GB
  - GPU GeForce GTX 1050Ti(4GB)  

### version_1　[深層学習による銀河形態分類Part1参照](http://satoshi-tomi.com/blog/galaxy/324/)
#### 銀河イメージデータのダウンロード (galaxy_getjpeg.ipynb)  
分類の対象とする銀河イメージデータのカタログとして、[Galaxy Zoo](https://data.galaxyzoo.org)の成果のひとつである"Galaxy Zoo 2"カタログ（[Willett et al.2013](http://arxiv.org/abs/1308.3496v2)、[Hart et al.(2016)](http://mnras.oxfordjournals.org/content/461/4/3663)参照）を使う。このカタログにはSDSS DR7のデータを中心に約30万個の銀河についての分類結果が記載されている。 "Galaxy Zoo 2"プロジェクトでは銀河形態をかなり細かいカテゴリに分類しているのだが、 とりあえず今回は銀河形態の最も基本的なカテゴリである「楕円銀河」(elliptical)と「渦巻銀河」(spiral)への分類を目指すことにし、"Galaxy Zoo 2"カタログ (<http://gz2hart.s3.amazonaws.com/gz2_hart16.fits.gz>) を「楕円銀河」カタログと「渦巻銀河」カタログに分割した。このプログラムでは、それぞれのカタログから最初の4000件の銀河イメージデータを、SDSSのWebサービスを利用してダウンロードしている。
#### 銀河形態分類詳細カテゴリ抽出 (galaxy_morphology_category.ipynb)  
[Galaxy Zoo](https://data.galaxyzoo.org/)では[ディシジョン・ツリー方式](https://data.galaxyzoo.org/gz_trees/gz_trees.html)によって形態カテゴリが決まるようになっている。実際にはどの程度のカテゴリに分類されているのかを調べるため、"Galaxy Zoo 2"カタログで使用されているすべての分類カテゴリを抽出してみた。  
#### Kerasによる銀河形態分類（2クラス）畳み込みニューラルネット (galaxcy_cnn_2class.ipynb)  
Chollet, F. (2017). *Deep Learning with Python.* Manning Publications Co. のChapter 5を参照してKerasのconv2Dによる畳み込みニューラルネットを実装し、「楕円銀河」、「渦巻銀河」合わせて4000件の訓練データセット、2000件の検証データセットに対してバッチサイズ20、エポック数30で訓練・検証を実行したところ、顕著な過学習に陥ることもなく、学習後は2000件のテストデータに対して98%の精度が得られた。

### version_2　[深層学習による銀河形態分類Part2参照](http://satoshi-tomi.com/blog/galaxy/353/)
#### 01_catalog
[GZ2カタログ](http://gz2hart.s3.amazonaws.com/gz2_hart16.fits.gz)と[GZ2メタデータ](http://zooniverse-data.s3.amazonaws.com/galaxy-zoo-2/gz2sample.fits.gz)から、１２クラス分類、8クラス分類、4クラス分類用のカタログを作成する。
- 12クラス分類用カタログ作成（galaxy_gz2_catalog_v2_12class.ipynb)
- 8クラス分類用カタログ作成（galaxy_gz2_catalog_v2_8class.ipynb)
- 4クラス分類用カタログ作成（galaxy_gz2_catalog_v2_4class.ipynb)

#### 02_datagen
[SDSS DR7のAPI](http://cas.sdss.org/dr7/en/help/docs/api.asp#Cutout)を使って、銀河イメージデータをjpeg形式でダウンロードし、訓練/検証/テスト用データセットを構成する。
- カタログに記載されているすべての銀河イメージデータ一括ダウンロードし、フルデータセットを作成(galaxy_getjpeg_all_gz2.ipynb)
- 各クラス指定件数分のランダムサンプリングダウンロード(galaxy_getjpeg_random_samples.ipynb)
- 指定された値以下の赤方偏移を持つデータを各クラス指定件数分ダウンロード(galaxy_getjpeg_redshift_samples.ipynb)
- 指定された値以下の光度を持つデータを各クラス指定件数分ダウンロード(galaxy_getjpeg_mr_samples.ipynb)
- フルデータセットから、各クラス4000件のランダムサンプリング・データセットを作成(galaxy_dataset_random_sampling.ipynb)
- フルデータセットから、各クラス4000件の赤方偏移優先サンプリング・データセットを作成(galaxy_dataset_selected_sampling.ipynb)
- フルデータセットから、赤方偏移&光度制限データセットを作成(galaxy_dataset_limited_sampling.ipynb)
- ランダムサンプリング・データセットまたは赤方偏移優先サンプリング・データセットを各クラス訓練用データ2000件、検証用データ1000件、テスト用データ1000件に分割(galaxy_imagedatagen_4000.ipynb)
- 赤方偏&光度制限データセットを各クラス訓練用データ400件、検証用データ400件、テスト用データ400件に分割(galaxy_imagedatagen_4000.ipynb)

#### 03_cnn
各データセットを対象としてKerasによる畳込みニューラルネットの訓練/検証を実行し、テスト用データを使用して精度を算出する。さらに、各クラスのテスト用データからランダムに10件のデータを抽出し、当該データのイメージとともに、各クラスへの予測確率を出力する。
- 12クラスのランダムサンプリング・データセットを対象とした畳込みニューラルネットの訓練/検証/テスト(galaxy_cnn_12class_256px_sf01_random_2000.ipynb)
- 12クラスの赤方偏移優先サンプリング・データセットを対象とした畳込みニューラルネットの訓練/検証/テスト(galaxy_cnn_12class_256px_sf01_selected_2000.ipynb)
- 12クラスの赤方偏移&光度制限データセットを対象とした畳込みニューラルネットの訓練/検証/テスト(galaxy_cnn_12class_256px_sf01_limited_400.ipynb)
- 8クラスのランダムサンプリング・データセットを対象とした畳込みニューラルネットの訓練/検証/テスト(galaxy_cnn_8class_256px_sf01_random_2000.ipynb)
- 8クラスの赤方偏移優先サンプリング・データセットを対象とした畳込みニューラルネットの訓練/検証/テスト(galaxy_cnn_8class_256px_sf01_selected_2000.ipynb)
- 8クラスの赤方偏移&光度制限データセットを対象とした畳込みニューラルネットの訓練/検証/テスト(galaxy_cnn_8class_256px_sf01_limited_400.ipynb)
- 4クラスのランダムサンプリング・データセットを対象とした畳込みニューラルネットの訓練/検証/テスト(galaxy_cnn_4class_256px_sf01_random_2000.ipynb)
- 4クラスの赤方偏移優先サンプリング・データセットを対象とした畳込みニューラルネットの訓練/検証/テスト(galaxy_cnn_4class_256px_sf01_selected_2000.ipynb)
- 4クラスの赤方偏移&光度制限データセットを対象とした畳込みニューラルネットの訓練/検証/テスト(galaxy_cnn_4class_256px_sf01_limited_400.ipynb)
- 12クラスのランダムサンプリング・データセットを対象とした転移学習畳込みニューラルネットの訓練/検証/テスト(galaxy_cnn_12class_transfer_256px_sf01_random_2000.ipynb)
- 12クラスの赤方偏移&光度制限データセットを対象とした転移学習畳込みニューラルネットの訓練/検証/テスト(galaxy_cnn_12class_transfer_256px_sf01_limited_400.ipynb)
