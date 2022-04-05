import tkinter
from tkinter import E, W, ttk
from tkinter import filedialog
from tkinter import messagebox
import mov2casa_GUI as casa
import pandas as pd
import os

def ask_movie_files():
    global movie_files
    movie_files = filedialog.askopenfilenames(initialdir=movie_dir)
    movie_path.set(os.path.dirname(movie_files[0]))
    movie_num['text'] = str(len(movie_files)) + ' files selected from'

def ask_result_folder():
    path = filedialog.askdirectory(initialdir=result_dir)
    result_path.set(path)

def ask_config_file():
    file = filedialog.askopenfilename()
    config_file.set(file)
    SetArg(file)

def do_analysis():  # 実行ボタンの処理
    input_dir = movie_path.get()
    output_dir = result_path.get()
    config_df = MakeArgDF()
    TestMode = 0
    if not input_dir or not output_dir:
        return
    casa.main(movie_files, output_dir, config_df, TestMode)
    messagebox.showinfo("完了", "完了しました。")

def make_testimg():  # テスト画像作成の実行
    input_dir = movie_path.get()
    output_dir = result_path.get()
    config_df = MakeArgDF()
    TestMode = 1
    if not input_dir or not output_dir:
        return
    casa.main(movie_files, output_dir, config_df, TestMode)
    messagebox.showinfo("完了", "完了しました。")

def SetArg(conf):
    cf = pd.read_csv(conf, header=None, sep=",")

    cnt_analysis_var.set(int(cf.loc[cf[0] == "cnt_analysis",1].values[0]))
    cropheight_var.set(int(cf.loc[cf[0] == "cropheight",1].values[0]))
    cropwidth_var.set(int(cf.loc[cf[0] == "cropwidth",1].values[0]))
    FrameRate_var.set(int(cf.loc[cf[0] == "FrameRate",1].values[0]))
    start_frame_var.set(int(cf.loc[cf[0] == "start_frame",1].values[0]))
    max_frame_var.set(int(cf.loc[cf[0] == "max_frame",1].values[0]))
    end_frame_var.set(int(cf.loc[cf[0] == "end_frame",1].values[0]))
    minsize_var.set(int(cf.loc[cf[0] == "minsize",1].values[0]))
    maxsize_var.set(int(cf.loc[cf[0] == "maxsize",1].values[0]))
    ovalratio_var.set(float(cf.loc[cf[0] == "ovalratio",1].values[0]))
    mobsearchrange_var.set(int(cf.loc[cf[0] == "mobsearchrange",1].values[0]))
    maxjumpframes_var.set(int(cf.loc[cf[0] == "maxjumpframes",1].values[0]))
    microscale_var.set(float(cf.loc[cf[0] == "microscale",1].values[0]))
    Threshtype_var.set(int(cf.loc[cf[0] == "Threshtype",1].values[0]))
    AThreshBS_var.set(int(cf.loc[cf[0] == "AThreshBS",1].values[0]))
    AThreshC_var.set(int(cf.loc[cf[0] == "AThreshC",1].values[0]))
    Bright_thresh_var.set(int(cf.loc[cf[0] == "Bright_thresh",1].values[0]))
    bright_erosion_iter_var.set(int(cf.loc[cf[0] == "bright_erosion_iter",1].values[0]))
    bright_dilate_iter_var.set(int(cf.loc[cf[0] == "bright_dilate_iter",1].values[0]))
    dark_erosion_iter_var.set(int(cf.loc[cf[0] == "dark_erosion_iter",1].values[0]))
    dark_dilate_iter_var.set(int(cf.loc[cf[0] == "dark_dilate_iter",1].values[0]))
    Motile_thresh_VSL_var.set(float(cf.loc[cf[0] == "Motile_thresh_VSL",1].values[0]))
    Motile_thresh_diameter_var.set(float(cf.loc[cf[0] == "Motile_thresh_diameter",1].values[0]))
    Progressive_var.set(int(cf.loc[cf[0] == "Progressive",1].values[0]))
    Circle_SumAngle_var.set(int(cf.loc[cf[0] == "Circle_SumAngle",1].values[0]))
    Circle_MeanAngle_var.set(int(cf.loc[cf[0] == "Circle_MeanAngle",1].values[0]))
    Circle_StdAngle_var.set(int(cf.loc[cf[0] == "Circle_StdAngle",1].values[0]))


def MakeArgDF():
    cnt_analysis = cnt_analysis_var.get()
    cropheight = cropheight_var.get()
    cropwidth = cropwidth_var.get()
    FrameRate = FrameRate_var.get()
    start_frame = start_frame_var.get()
    max_frame = max_frame_var.get()
    end_frame = end_frame_var.get()
    minsize = minsize_var.get()
    maxsize = maxsize_var.get()
    ovalratio = ovalratio_var.get()
    mobsearchrange = mobsearchrange_var.get()
    maxjumpframes = maxjumpframes_var.get()
    microscale = microscale_var.get()
    Threshtype = Threshtype_var.get()
    AThreshBS = AThreshBS_var.get()
    AThreshC = AThreshC_var.get()
    Bright_thresh = Bright_thresh_var.get()
    bright_erosion_iter = bright_erosion_iter_var.get()
    bright_dilate_iter = bright_dilate_iter_var.get()
    dark_erosion_iter = dark_erosion_iter_var.get()
    dark_dilate_iter = dark_dilate_iter_var.get()
    Motile_thresh_VSL = Motile_thresh_VSL_var.get()
    Motile_thresh_diameter = Motile_thresh_diameter_var.get()
    Progressive = Progressive_var.get()
    Circle_SumAngle = Circle_SumAngle_var.get()
    Circle_MeanAngle = Circle_MeanAngle_var.get()
    Circle_StdAngle = Circle_StdAngle_var.get()

    df = pd.DataFrame(
        data={0:['cnt_analysis','cropheight','cropwidth','FrameRate','start_frame','max_frame','end_frame','minsize','maxsize','ovalratio','mobsearchrange','maxjumpframes','microscale','Threshtype','AThreshBS','AThreshC','Bright_thresh','bright_erosion_iter','bright_dilate_iter','dark_erosion_iter','dark_dilate_iter','Motile_thresh_VSL','Motile_thresh_diameter','Progressive','Circle_SumAngle','Circle_MeanAngle','Circle_StdAngle'],
            1:[cnt_analysis,cropheight,cropwidth,FrameRate,start_frame,max_frame,end_frame,minsize,maxsize,ovalratio,mobsearchrange,maxjumpframes,microscale,Threshtype,AThreshBS,AThreshC,Bright_thresh,bright_erosion_iter,bright_dilate_iter,dark_erosion_iter,dark_dilate_iter,Motile_thresh_VSL,Motile_thresh_diameter,Progressive,Circle_SumAngle,Circle_MeanAngle,Circle_StdAngle]
            })

    return df

def SaveArgfile():
    df = MakeArgDF()

    savefilename = filedialog.asksaveasfilename(
        title = '設定ファイルの保存',
        filetype = [('CSV', '.csv')],
        initialdir = './',
        defaultextension = 'csv'
    )
    config_file.set(savefilename)

    df.to_csv(savefilename, header=False, index=False)

movie_dir = 'blank'
result_dir = ''
config_path = ''

def ReadPathHist():
    f = open('./path_hist.txt')
    movie_dir = f.readline()
    result_dir = f.readline()
    config_path = f.readline()

    return movie_dir, result_dir, config_path


# メインウィンドウ
main_win = tkinter.Tk()
main_win.title("Sperm Motility Analyzer")
main_win.geometry("800x450")

# 動画指定フレーム
movie_frm = ttk.Frame(main_win)
movie_frm.grid(column=0, row=0, padx=5)

# メインフレーム
main_frm = ttk.Frame(main_win)
main_frm.grid(column=0, row=1, padx=5)

# 設定フレーム
conf_frm = ttk.Frame(main_win)
conf_frm.grid(column=0, row=2, padx=5, pady=10)

# ボタンフレーム
btn_frm = ttk.Frame(main_win)
btn_frm.grid(column=0, row=3, padx=5, pady=10)



# パラメータ
movie_path = tkinter.StringVar()
result_path = tkinter.StringVar()
config_file = tkinter.StringVar()

# ウィジェット（動画フォルダ名）
movie_label1 = ttk.Label(movie_frm, width=25, text="動画ファイル指定（複数可）")
movie_num = ttk.Label(movie_frm, width=25, text='0 files selected from')
movie_box1 = ttk.Entry(movie_frm, width=55, textvariable=movie_path)
movie_btn1 = ttk.Button(movie_frm, width=10, text='選択', command=ask_movie_files)

# ウィジェット（結果フォルダ名）
result_label2 = ttk.Label(main_frm, width=25, text="結果出力フォルダ指定")
result_box2 = ttk.Entry(main_frm, width=80, textvariable=result_path)
result_btn2 = ttk.Button(main_frm, width=10, text="参照", command=ask_result_folder)

# ウィジェット（設定ファイル名）
file_label = ttk.Label(main_frm, width=25, text="設定ファイル")
file_box = ttk.Entry(main_frm, width=80, textvariable=config_file)
file_btn = ttk.Button(main_frm, width=10, text="設定読込", command=ask_config_file)

# 設定保存ボタン
save_config_btn = ttk.Button(conf_frm, text='名前をつけて設定を保存', command=SaveArgfile)

# 実行ボタン
app_btn = ttk.Button(btn_frm, text="解析実行", command=do_analysis)

# テスト画像生成ボタン
test_btn = ttk.Button(btn_frm, text="テスト画像作成", command=make_testimg)


# 設定項目
cnt_analysis_var = tkinter.StringVar()
cnt_analysis_label = ttk.Label(conf_frm, width=30, text='希望する最大解析精子数')
cnt_analysis_box = ttk.Entry(conf_frm, textvariable=cnt_analysis_var, width=8)
cropheight_var = tkinter.StringVar()
cropheight_label = ttk.Label(conf_frm, width=30, text='画角切り取り高')
cropheight_box = ttk.Entry(conf_frm, textvariable=cropheight_var, width=8)
cropwidth_var = tkinter.StringVar()
cropwidth_label = ttk.Label(conf_frm, width=30, text='画角切り取り幅')
cropwidth_box = ttk.Entry(conf_frm, textvariable=cropwidth_var, width=8)
FrameRate_var = tkinter.StringVar()
FrameRate_label = ttk.Label(conf_frm, width=30, text='動画のフレームレート')
FrameRate_box = ttk.Entry(conf_frm, textvariable=FrameRate_var, width=8)
start_frame_var = tkinter.StringVar()
start_frame_label = ttk.Label(conf_frm, width=30, text='解析開始フレーム')
start_frame_box = ttk.Entry(conf_frm, textvariable=start_frame_var, width=8)
max_frame_var = tkinter.StringVar()
max_frame_label = ttk.Label(conf_frm, width=30, text='解析に用いるフレーム数')
max_frame_box = ttk.Entry(conf_frm, textvariable=max_frame_var, width=8)
end_frame_var = tkinter.StringVar()
end_frame_label = ttk.Label(conf_frm, width=30, text='解析終了フレーム（0:全フレーム）')
end_frame_box = ttk.Entry(conf_frm, textvariable=end_frame_var, width=8)
minsize_var = tkinter.StringVar()
minsize_label = ttk.Label(conf_frm, width=30, text='精子と認識する最小面積')
minsize_box = ttk.Entry(conf_frm, textvariable=minsize_var, width=8)
maxsize_var = tkinter.StringVar()
maxsize_label = ttk.Label(conf_frm, width=30, text='精子と認識する最大面積')
maxsize_box = ttk.Entry(conf_frm, textvariable=maxsize_var, width=8)
ovalratio_var = tkinter.StringVar()
ovalratio_label = ttk.Label(conf_frm, width=30, text='精子頭部と認識する縦横比率')
ovalratio_box = ttk.Entry(conf_frm, textvariable=ovalratio_var, width=8)
mobsearchrange_var = tkinter.StringVar()
mobsearchrange_label = ttk.Label(conf_frm, width=30, text='連結精子を探す範囲')
mobsearchrange_box = ttk.Entry(conf_frm, textvariable=mobsearchrange_var, width=8)
maxjumpframes_var = tkinter.StringVar()
maxjumpframes_label = ttk.Label(conf_frm, width=30, text='このフレーム数先まで探す')
maxjumpframes_box = ttk.Entry(conf_frm, textvariable=maxjumpframes_var, width=8)
microscale_var = tkinter.StringVar()
microscale_label = ttk.Label(conf_frm, width=30, text='Pixel x この数値 = 実寸')
microscale_box = ttk.Entry(conf_frm, textvariable=microscale_var, width=8)
Threshtype_var = tkinter.StringVar()
Threshtype_label = ttk.Label(conf_frm, width=30, text='二値化の方法')
Threshtype_box = ttk.Entry(conf_frm, textvariable=Threshtype_var, width=8)
AThreshBS_var = tkinter.StringVar()
AThreshBS_label = ttk.Label(conf_frm, width=30, text='適応的二値化のサイズ')
AThreshBS_box = ttk.Entry(conf_frm, textvariable=AThreshBS_var, width=8)
AThreshC_var = tkinter.StringVar()
AThreshC_label = ttk.Label(conf_frm, width=30, text='適応的二値化の閾値')
AThreshC_box = ttk.Entry(conf_frm, textvariable=AThreshC_var, width=8)
Bright_thresh_var = tkinter.StringVar()
Bright_thresh_label = ttk.Label(conf_frm, width=30, text='明部認識の閾値')
Bright_thresh_box = ttk.Entry(conf_frm, textvariable=Bright_thresh_var, width=8)
bright_erosion_iter_var = tkinter.StringVar()
bright_erosion_iter_label = ttk.Label(conf_frm, width=30, text='明部二値化収縮処理の回数')
bright_erosion_iter_box = ttk.Entry(conf_frm, textvariable=bright_erosion_iter_var, width=8)
bright_dilate_iter_var = tkinter.StringVar()
bright_dilate_iter_label = ttk.Label(conf_frm, width=30, text='明部二値化膨張処理の回数')
bright_dilate_iter_box = ttk.Entry(conf_frm, textvariable=bright_dilate_iter_var, width=8)
dark_erosion_iter_var = tkinter.StringVar()
dark_erosion_iter_label = ttk.Label(conf_frm, width=30, text='暗部二値化収縮処理の回数')
dark_erosion_iter_box = ttk.Entry(conf_frm, textvariable=dark_erosion_iter_var, width=8)
dark_dilate_iter_var = tkinter.StringVar()
dark_dilate_iter_label = ttk.Label(conf_frm, width=30, text='暗部二値化膨張処理の回数')
dark_dilate_iter_box = ttk.Entry(conf_frm, textvariable=dark_dilate_iter_var, width=8)
Motile_thresh_VSL_var = tkinter.StringVar()
Motile_thresh_VSL_label = ttk.Label(conf_frm, width=30, text='運動精子と判定する最小VSL')
Motile_thresh_VSL_box = ttk.Entry(conf_frm, textvariable=Motile_thresh_VSL_var, width=8)
Motile_thresh_diameter_var = tkinter.StringVar()
Motile_thresh_diameter_label = ttk.Label(conf_frm, width=30, text='運動精子と判定する最小点間')
Motile_thresh_diameter_box = ttk.Entry(conf_frm, textvariable=Motile_thresh_diameter_var, width=8)
Progressive_var = tkinter.StringVar()
Progressive_label = ttk.Label(conf_frm, width=30, text='前進性の高い精子と判定するVAP')
Progressive_box = ttk.Entry(conf_frm, textvariable=Progressive_var, width=8)
Circle_SumAngle_var = tkinter.StringVar()
Circle_SumAngle_label = ttk.Label(conf_frm, width=30, text='旋回精子と判定する角度合計')
Circle_SumAngle_box = ttk.Entry(conf_frm, textvariable=Circle_SumAngle_var, width=8)
Circle_MeanAngle_var = tkinter.StringVar()
Circle_MeanAngle_label = ttk.Label(conf_frm, width=30, text='旋回精子と判定する角度平均')
Circle_MeanAngle_box = ttk.Entry(conf_frm, textvariable=Circle_MeanAngle_var, width=8)
Circle_StdAngle_var = tkinter.StringVar()
Circle_StdAngle_label = ttk.Label(conf_frm, width=30, text='旋回精子と判定する角度標準偏差')
Circle_StdAngle_box = ttk.Entry(conf_frm, textvariable=Circle_StdAngle_var, width=8)



# ウィジェットの配置
movie_label1.grid(column=0, row=0, sticky=W)
movie_num.grid(column=1, row=0, sticky=tkinter.E)
movie_box1.grid(column=2, row=0, sticky=tkinter.EW, pady=2)
movie_btn1.grid(column=3, row=0, padx=5)
result_label2.grid(column=0, row=1, sticky=W)
result_box2.grid(column=1, row=1, sticky=tkinter.EW, pady=2)
result_btn2.grid(column=2, row=1, padx=5)
file_label.grid(column=0, row=2, sticky=W)
file_box.grid(column=1, row=2, sticky=tkinter.EW, pady=2)
file_btn.grid(column=2, row=2, padx=5)

cnt_analysis_label.grid(column=0, row=3, sticky=W, padx=5, pady=2)
cnt_analysis_box.grid(column=1, row=3)
cropheight_label.grid(column=0, row=4, sticky=W, padx=5, pady=2)
cropheight_box.grid(column=1, row=4)
cropwidth_label.grid(column=0, row=5, sticky=W, padx=5, pady=2)
cropwidth_box.grid(column=1, row=5)
FrameRate_label.grid(column=0, row=6, sticky=W, padx=5, pady=2)
FrameRate_box.grid(column=1, row=6)
start_frame_label.grid(column=0, row=7, sticky=W, padx=5, pady=2)
start_frame_box.grid(column=1, row=7)
max_frame_label.grid(column=0, row=8, sticky=W, padx=5, pady=2)
max_frame_box.grid(column=1, row=8)
end_frame_label.grid(column=0, row=9, sticky=W, padx=5, pady=2)
end_frame_box.grid(column=1, row=9)
minsize_label.grid(column=0, row=10, sticky=W, padx=5, pady=2)
minsize_box.grid(column=1, row=10)
maxsize_label.grid(column=0, row=11, sticky=W, padx=5, pady=2)
maxsize_box.grid(column=1, row=11)
ovalratio_label.grid(column=0, row=12, sticky=W, padx=5, pady=2)
ovalratio_box.grid(column=1, row=12)

mobsearchrange_label.grid(column=2, row=3, sticky=W, padx=5, pady=2)
mobsearchrange_box.grid(column=3, row=3)
maxjumpframes_label.grid(column=2, row=4, sticky=W, padx=5, pady=2)
maxjumpframes_box.grid(column=3, row=4)
microscale_label.grid(column=2, row=5, sticky=W, padx=5, pady=2)
microscale_box.grid(column=3, row=5)
Motile_thresh_VSL_label.grid(column=2, row=6, sticky=W, padx=5, pady=2)
Motile_thresh_VSL_box.grid(column=3, row=6)
Motile_thresh_diameter_label.grid(column=2, row=7, sticky=W, padx=5, pady=2)
Motile_thresh_diameter_box.grid(column=3, row=7)
Progressive_label.grid(column=2, row=8, sticky=W, padx=5, pady=2)
Progressive_box.grid(column=3, row=8)
Circle_SumAngle_label.grid(column=2, row=9, sticky=W, padx=5, pady=2)
Circle_SumAngle_box.grid(column=3, row=9)
Circle_MeanAngle_label.grid(column=2, row=10, sticky=W, padx=5, pady=2)
Circle_MeanAngle_box.grid(column=3, row=10)
Circle_StdAngle_label.grid(column=2, row=11, sticky=W, padx=5, pady=2)
Circle_StdAngle_box.grid(column=3, row=11)

Threshtype_label.grid(column=4, row=3, sticky=W, padx=5, pady=2)
Threshtype_box.grid(column=5, row=3)
AThreshBS_label.grid(column=4, row=4, sticky=W, padx=5, pady=2)
AThreshBS_box.grid(column=5, row=4)
AThreshC_label.grid(column=4, row=5, sticky=W, padx=5, pady=2)
AThreshC_box.grid(column=5, row=5)
Bright_thresh_label.grid(column=4, row=6, sticky=W, padx=5, pady=2)
Bright_thresh_box.grid(column=5, row=6)
bright_erosion_iter_label.grid(column=4, row=7, sticky=W, padx=5, pady=2)
bright_erosion_iter_box.grid(column=5, row=7)
bright_dilate_iter_label.grid(column=4, row=8, sticky=W, padx=5, pady=2)
bright_dilate_iter_box.grid(column=5, row=8)
dark_erosion_iter_label.grid(column=4, row=9, sticky=W, padx=5, pady=2)
dark_erosion_iter_box.grid(column=5, row=9)
dark_dilate_iter_label.grid(column=4, row=10, sticky=W, padx=5, pady=2)
dark_dilate_iter_box.grid(column=5, row=10)

save_config_btn.grid(column=4, row=12, sticky=tkinter.EW)


app_btn.grid(column=0, row=13, sticky=tkinter.EW)
test_btn.grid(column=1, row=13, sticky=tkinter.EW)


# 配置設定
main_win.columnconfigure(0, weight=1)
main_win.rowconfigure(0, weight=1)
movie_frm.columnconfigure(4, weight=1)
main_frm.columnconfigure(1, weight=1)


# 処理
initial_config_path = './config_GUI.csv'
config_file.set(initial_config_path)
SetArg(initial_config_path)
movie_dir, result_dir, config_path = ReadPathHist()
#movie_path.set(movie_dir)
#result_path.set(result_dir)
#config_file.set(config_path)


main_win.mainloop()