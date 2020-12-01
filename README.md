# sperm_motility_analyzer
精子運動性解析

CASAシステムマニュアル
2019.6.21
環境のインストール
Python3の環境がすでにインストールされている場合
以下のライブラリが必要ですので、インストールしてください。
numpy
pandas
OpenCV3
tqdm
PIL

Python環境をゼロからインストールする場合（Anacondaのインストール）
プログラミング言語Pythonと、数値解析用ライブラリ等一式をまとめて導入できるAnacondaをインストールします。
https://www.anaconda.com
右上のDownloadボタンからダウンロードページに行き、Python 3.* version をダウンロードし、インストールしてください。
（64Bitか32Bitかは、使用するPCによります)
【Windowsの場合】Advanced Option の Add Anaconda to my PATH environment variable にチェックを入れてください。
インストールが終了したら、PCを再起動してください。
その他必要なライブラリのインストール
OpenCV3
OpenCV3は画像解析用ライブラリです。
【Windowsの場合】コマンドプロンプト、【Macの場合】ターミナルからインストール操作を行います。
【Windowsの場合】Windowsのコマンドプロンプトを以下の通り管理者モードで開いてください。
	・コマンドプロンプトのアイコンを出す。
	　　・Windows10 の場合、スタートメニューの「Windowsシステムツール」内
	・右クリック→その他→管理者として実行。
【Macの場合】管理者権限のあるユーザーでログインしていれば、ターミナルで以下の操作ができます。
以下のコマンドを入力してください。
pip install opencv-python
conda config --add channels conda-forge
conda install opencv
途中 Proceed([y]/n)? と聞かれるので、yを押してください。
インストールが終了したら、PCを再起動してください。
＊2019.4.7 上記方法でインストールされるバージョンがOpenCV4になったかもしれません。OpenCV4対応版をご使用ください。
tqdm
＊AnacondaのPython3.7以上をインストールした場合は、すでにインストールされているため、この操作は不要です。
tqdmは処理の進行状況を示すバーを表示します。
コマンドプロンプトを管理者モードで開き、以下のコマンドを入力してください。
pip install tqdm
CASAシステムのインストール
CASA_SETフォルダを適当なところに置いてください。
 
設定ファイル"config.csv"
config.csvは各種設定を行うファイルです。使用する顕微鏡やカメラに合わせて設定してください。
TestMode
1にするとテストモードになります。動画0フレーム目の二値化画像に認識結果を重ねた画像を生成します。
MovieFolder
解析する動画を入れるフォルダ名の設定。”./” と”/” の間にフォルダ名を入れてください。
ResultFolder
結果を出力するフォルダ名を設定。”./” と”/” の間にフォルダ名を入れてください。
cropheight
動画の解析範囲を制限する場合に指定。動画中心部の指定の高さの範囲を切り出します。0は切り出しなし。
cropwidth
動画の解析範囲を制限する場合に指定。動画中心部の指定の幅の範囲を切り出します。0は切り出しなし。
start_second
指定した秒数後から1秒間の動画を解析します。
FrameRate
フレームレートの指定。
minsize
精子として認識する最小サイズ（ピクセル数）。
maxsize
精子として認識する最大サイズ（ピクセル数）。
maxjumpframes
最大何フレーム先まで探すか。
mobsearchrange
運動精子の次フレームの移動先を探す範囲。 現フレームの運動精子の位置から距離mobsearchrange以内の次フレームの運動精子のうち、最も近い精子を同一精子とする。
ovalratio
球体（泡や精子以外の細胞など）と精子頭部（楕円）を見分けるため、認識された粒子の面積が、粒子の最小外接円 × ovalratio 以下の場合に精子と認定する。
microscale
ピクセル数と実サイズの比率。（実サイズ＝ピクセル数 × microscale μm）
Threshtype
二値化の方法を選択する。
0は大津の二値化。（一つの閾値が自動で算出される。）
1は適応的二値化。（画像の小領域ごとに閾値が計算される。）
2~255は、入力値を閾値とした二値化を行う。
AThreshBS
適応的二値化のための解析範囲。
AThreshC
適応的二値化の重み。
bright_erosion_iter
明部収縮処理の回数。
bright_dilate_iter
明部膨張処理の回数。
dark_erosion_iter
暗部収縮処理の回数。
dark_dilate_iter
暗部膨張処理の回数。
MaskThreshold
不動精子マスク画像を作成する閾値。
Motile_thresh_diameter
精子の軌跡中の最も遠い点間の距離がこの値以下だった場合、不動精子と認定する。
Progressive
VAPがこの値（μm）以上の時に、前進性の高い精子と判定する。
Circle_SumAngle
SumAngleがこの値以上であり、
Circle_MeanAngle
MeanAngleがこの値以上であり、
Circle_StdAngle
StdAngleがこの値以下であった場合、円運動をしていると判定する。
Derailed_StdAngle
StdAngleがこの値以上だった場合、軌跡がずれた（脱線）と判定する。
（試行錯誤中。誤って脱線と判定されないように十分大きな値にしてください。）


config.csvは、テキストエディタ等での編集を推奨します。
Excelで編集して保存すると、読み取れなくなる可能性があります。
その場合は、一度メモ帳でconfig.csvを開き、「ファイル―名前を付けて保存」で出てくる保存画面の下部「文字コード」のところを「UTF-8」にして、保存しなおしてください。
 
解析の仕組み
精子の検出
動画の1秒分（FrameRateで指定：推奨60フレーム/秒以上）の画像を解析します。動画は明視野の画像（精子が黒く見える）で結構です。
画像はグレースケールで読み込み、1枚1枚に対して二値化処理を行い、白黒反転して二値の画像に変換します。これにはOpenCVのAdaptive Threshold（適応的二値化）を用いています。
牛のように精子頭部が扁平である場合、精子の頭部に光が反射して白くなる＝二値化で検出できなくなることがあります。これに対処するため、画像中の極端に明るい部分を検出して二値化し、上記の適応的二値化の白黒反転画像と重ねることにより、黒い部分と光る部分を両方検出できるようにしています。こうして、黒い背景に精子が白く映った画像が作成されます。
明部と暗部の二値化画像それぞれに対して、ノイズを消去するため「収縮処理」と「膨張処理」を行うことができます。収縮処理を行うと、精子の輪郭が削られ小さくなりますが、極小ドットのノイズを消すことができます。膨張処理は、精子が小さくなりすぎた場合に輪郭を戻す操作です。膨張処理を行っても、ノイズは消えたままです。収縮処理の実施はbright_erosion_iter（明部）、dark_erosion_iter（暗部）で回数を指定します。膨張処理の実施はbright_dilate_iter（明部）、dark_ dilate _iter（暗部）で回数を指定します。
このように加工した二値化画像から、OpenCVの機能を用いて粒子を検出します。粒子の面積がminsize以上、maxsize以下であり、粒子が円形でない（粒子の外接円の面積×ovalratio > 粒子の面積）ものを精子頭部として認識します。
不動精子と運動精子の区別
不動精子と運動精子が重なった場合、運動精子の軌跡が途切れることがあります。これを防ぐため、「不動精子マスク」を作成し、元の画像から差し引くことにより、運動精子だけを検出するようにしています。
不動精子マスクは、1秒分の二値化画像を重ねた平均をとり、MaskThresholdを閾値として二値化した画像です。二値化画像を重ねた平均をとると、動かない精子ほど同じ位置で重なって「濃く」なります。この濃い部分を残すように二値化することにより、不動精子のみが映ったマスク画像が作成できます。
軌跡の連結
0フレーム目に存在する運動精子1個1個について、次フレームの最も近い位置の粒子を探します。逆に、その粒子にとって前フレームの最も近い位置の粒子が当該運動精子と一致した場合、同じ運動精子であると認定します。次フレームのmobsearchrangeで設定した範囲内に粒子が見られなかった場合は、1フレーム飛ばした先を探します。これをmaxjumpframesで設定したフレーム数まで繰り返します。以上の操作を0フレームからFrameRateで指定したフレーム数まで繰り返して軌跡を連結します。
運動精子の認定
軌跡の頂点の最も遠い2点の距離がMotile_thresh_diameter以上の精子を運動精子と認定します。運動精子は軌跡が途切れた場合でも認定されます。0フレーム目の画像内で精子と認定されたものについて、運動精子/全精子を運動精子率としています。
VAPがProgressiveで指定した値以上の精子は、前進性の高い精子と判定します。ここでは軌跡が途切れた場合でも、それまでのVAPを維持して最終フレームまで動いたと換算して評価します。
他の解析項目は、0〜FrameRateまで軌跡がつながっているものだけについて集計しています。解析中に画面外に出たもの、解析中に画面外から入ってきたもの、軌跡が途切れたものは集計対象外です。
平均経路の算出
フレームレートの1/6個分前からの座標の平均値を、x, yそれぞれ算出し、平均経路の座標としています。始点付近では平均する個数を1個ずつ増やし、終点に近いところでは平均する個数を1個ずつ減らすことにより、平均経路の始点と終点が軌跡の始点と終点に一致するようにしています。平均経路を結んだ距離が、VAPです。
円運動精子の検出
平均経路で3フレームごとに、経路が左右にずれる角度を＋/－で算出しています。これらをすべて合計した値の絶対値がSumAngle、平均の絶対値がMeanAngleです。これらの値が大きいとき、精子が円運動をしていると考えられます。ただし、角度の標準偏差StdAngleが大きい場合、円運動でなく経路が急に折れ曲がっただけである可能性があります。
フラクタル次元の算出
フラクタル次元（D）は軌跡の複雑さを表す値です。直線は1次元、平面は2次元ですが、フラクタル次元ではジグザグした軌跡を1次元と2次元の間と考えます。平面を塗りつぶすような密な軌跡を描くほど2次元に近くなります。（同じ個所を何度も通るような場合は、2次元を超える場合もあります。）
D = log(n)/(log(n) + log(d/L)) で算出されます。nは頂点の間隔の数、dは最も遠い頂点同士の距離、Lは軌跡の長さを表します。　
 
解析のしかた
解析する動画の保存
動画はMovieFolderで設定したフォルダにまとめて入れてください。
設定の最適化
最初は使用機器にあわせて、試行錯誤しながら設定を最適化する必要があります。
まず、config.csvのTestModeを1にして、精子が正しく認識されるようにパラメーターを調整してください。
その後、TestModeを0にして動画解析し、正確に精子の軌跡を追えるように調整してください。
解析
【Windowsの場合】casa.batをダブルクリックします。
バッチファイルが使用できない場合（Macなど）、コマンドプロンプトやターミナルでカレントディレクトリをmov2casa.pyのあるフォルダに移動し、
・python mov2casa.py
と打ち込んでください。
解析は自動で進行し、ResultFolderに設定したフォルダに結果が出力されます。

解析結果
ファイル名_detect.jpg
テストモードで生成される画像です。0フレーム目の画像と、二値化画像に精子として認識された座標を赤点で重ねたものを連結しています。
ファイル名_mask.jpg
テストモードで生成される画像です。不動精子マスク画像を出力しています。
ファイル名_AllPoints.csv
解析した全精子の座標が記録されています。
Frame：フレーム数
x, y, area：精子の座標、および面積。
Mov：0フレーム時点での精子の運動性認定。不動精子：０、運動精子：１
Point：精子の番号。同じ番号は同一精子と認識されたことを示します。
Ave_x, Ave_y：平均座標
Runlength：0フレームからの移動距離積算
Framelength：フレーム数に同じ
Velocity：精子の速度（Runlength/Framelength）
fix：精子の情報が確定されたかどうか（解析時使用パラメーター）
angle：VAPの3フレームごとに、軌跡が左右に曲がる角度を算出します。
motile：生存精子は1
derail ：軌跡が他と入れ替わるなどの異常がある可能性があるものは１（検討中）
prog：前進性の高い精子は１
circle：円運動している精子は１
ファイル名_CASA.csv
精子ごとに集計した解析値が記録されています。
FL_VCL, FL_VAP：VCLおよびVAPを算出したフレーム長
SumAngle：角度の合計。大きいほど精子が旋回運動をしていることを示す。
MeanAngle：角度の平均。０に近ければ直線運動に近い。大きいほど旋回運動が急であることを示す。
StdAngle：角度の標準偏差。大きい場合、角度のばらつきが大きい。旋回運動でない可能性がある。
diameter：精子の軌跡を内包する外接円の直径。最も遠い二点間の距離を表す。これが一定以上の精子を運動精子と認定する。
D：フラクタル次元。軌跡の密度、複雑さを表す。
ファイル名_track.m4v
精子運動の軌跡を上書きした動画です。
軌跡は空色で示し、不動精子は青い数字、運動精子のうち前進性の高い精子は赤い数字、前進性が閾値以下の精子は黄色い数字で表しています。数字はPointの値と一致します。
ファイル名_lastframe.jpg
軌跡を上書きした動画の最終フレームだけの画像です。
AllResult.csv
同時に解析した動画について、動画ごとの運動精子率、前進性の高い精子率、円運動精子率、解析項目中央値が集計されています。
解析項目については、集計は0~最終フレームまでデータがある精子のみで行います。
No. Analyzed：解析項目の集計に用いられた運動精子の数。
0F_total：0フレームに存在する全精子数。
Motility：運動精子率。0フレーム時点での運動精子数/全精子数。
Prog_Rate：前進性の高い精子率（0フレーム時点）。
Circle_Rate：円運動と判定された精子率（0フレーム時点）。Motility，Prog_Rateと同じく、軌跡が途切れた精子も評価されますが、軌跡が早くに途切れた場合は円運動していてもSumAngleが閾値Circle_SumAngleを超えず、認定されない場合があります。
 
解析項目の解説
VCL（Velocity Curbilinear：曲線速度）
粒子の重心座標を直接結んだ距離
VAP（Velocity Average Path：移動平均速度）
平均座標（10フレーム分）を結んだ距離
VSL（Velocity Straight Line：直線距離）
VAPの始点から終点までの距離
LIN（Linearity：直進性）
直進性の度合い。（VSL/VCL）
STR（Straightness：直線性）
直線性の度合い。（VSL/VAP）
WOB（Wobble：ふらつき）
精子頭部が左右に振れる度合い。（VAP/VCL）
BCF（Beat-cross frequency：振幅周波数）
VCLがVAPと交差する数
ALH（Amplitude of lateral head displacement）
頭部横移動の大きさ平均（推定値）
D（フラクタル次元）
軌跡の複雑さを表す。
